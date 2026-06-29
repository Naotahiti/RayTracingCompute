
#include "vulkan/vulkan.h"
#include "glm/glm.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>


struct CameraParamsGPU {
    glm::vec3 center;            float _pad0;
    glm::vec3 pixel00_loc;       float _pad1;
    glm::vec3 pixel_delta_u;     float _pad2;
    glm::vec3 pixel_delta_v;     float _pad3;
    glm::vec3 u;                 float _pad4;
    glm::vec3 v;                 float _pad5;
    glm::vec3 w;                 float _pad6;
    glm::vec3 defocus_disk_u;    float _pad7;
    glm::vec3 defocus_disk_v;    float _pad8;
    int       image_width;
    int       image_height;
    int       samples_per_pixel;
    int       max_depth;
    float     defocus_angle;
    int       sphere_count;
    int       _pad9;
    int       _pad10;
};

struct SphereGPU {
    glm::vec3 center;  float radius;
    int       materialIndex;
    float     _pad0, _pad1, _pad2;
};

struct MaterialGPU {
    glm::vec3 albedo;
    float     fuzz;
    float     refraction_index;
    int       type; // 0 = lambertian, 1 = metal, 2 = dielectric
    float     _pad0, _pad1;
};

double random_double() {
    return std::rand() / (RAND_MAX + 1.0);
}
double random_double(double min, double max) {
    return min + (max - min) * random_double();
}


class VulkanCompute {
public:
    void run(int image_width, int image_height, int samples_per_pixel, int max_depth,
        glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 vup,
        double vfov, double defocus_angle, double focus_dist,
        const std::vector<SphereGPU>& sphereList,
        const std::vector<MaterialGPU>& materialList)
    {
        this->image_width = image_width;
        this->image_height = image_height;

        createInstance();
        pickPhysicalDevice();
        createLogicalDeviceAndQueue();
        createCommandPool();

        CameraParamsGPU cam = computeCameraParams(image_width, image_height, samples_per_pixel,
            max_depth, lookfrom, lookat, vup, vfov, defocus_angle, focus_dist,
            (int)sphereList.size());

        VkDeviceSize outputSize = sizeof(glm::vec4) * image_width * image_height;
        VkDeviceSize uniformSize = sizeof(CameraParamsGPU);
        VkDeviceSize sphereSize = sizeof(SphereGPU) * sphereList.size();
        VkDeviceSize materialSize = sizeof(MaterialGPU) * materialList.size();

        createBuffer(outputSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            outputBuffer, outputMemory);

        createBuffer(uniformSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffer, uniformMemory);

        createBuffer(sphereSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            sphereBuffer, sphereMemory);

        createBuffer(materialSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            materialBuffer, materialMemory);

        copyToBuffer(uniformMemory, &cam, uniformSize);
        copyToBuffer(sphereMemory, sphereList.data(), sphereSize);
        copyToBuffer(materialMemory, materialList.data(), materialSize);

        createDescriptorSetLayout();
        createComputePipeline("raytrace.spv");
        createDescriptorPoolAndSet();
        recordAndSubmit(image_width, image_height);

        writePPM("image.ppm", image_width, image_height);

        cleanup();
    }

private:
    VkInstance instance{};
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    VkQueue computeQueue{};
    uint32_t queueFamilyIndex = 0;

    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};

    VkBuffer outputBuffer{}, uniformBuffer{}, sphereBuffer{}, materialBuffer{};
    VkDeviceMemory outputMemory{}, uniformMemory{}, sphereMemory{}, materialMemory{};

    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline computePipeline{};
    VkDescriptorPool descriptorPool{};
    VkDescriptorSet descriptorSet{};

    int image_width = 0, image_height = 0;

    void createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RT_Compute";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Echec creation instance Vulkan");
}
    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("Aucun GPU compatible Vulkan trouve");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        physicalDevice = devices[0];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "GPU utilise : " << props.deviceName << "\n";
    }

    void createLogicalDeviceAndQueue() {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());

        for (uint32_t i = 0; i < count; i++) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;

        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Echec creation device logique");

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Echec creation command pool");
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Type de memoire GPU compatible introuvable");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory) {

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Echec creation buffer");

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Echec allocation memoire GPU");

        vkBindBufferMemory(device, buffer, memory, 0);
    }
    void copyToBuffer(VkDeviceMemory memory, const void* src, VkDeviceSize size) {
        if (size == 0) return; // evite un memcpy de taille 0 (cas scene vide)
        void* dst;
        vkMapMemory(device, memory, 0, size, 0, &dst);
        memcpy(dst, src, (size_t)size);
        vkUnmapMemory(device, memory);
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[4]{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Echec creation descriptor set layout");
    }

    std::vector<char> readSPV(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Impossible d'ouvrir le fichier shader compile : " + filename);

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }

    void createComputePipeline(const std::string& spvPath) {
        auto code = readSPV(spvPath);

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size();
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("Echec creation shader module");

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Echec creation pipeline layout");

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS)
            throw std::runtime_error("Echec creation compute pipeline");

        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    void createDescriptorPoolAndSet() {
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 3; // output + sphere + material
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Echec creation descriptor pool");

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("Echec allocation descriptor set");

        VkDescriptorBufferInfo outputInfo{ outputBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo uniformInfo{ uniformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo sphereInfo{ sphereBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo materialInfo{ materialBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[4]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &outputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo = &uniformInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &sphereInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = descriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &materialInfo;

        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }


    void recordAndSubmit(int width, int height) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        uint32_t groupsX = (width + 7) / 8;
        uint32_t groupsY = (height + 7) / 8;
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue);
    }

    CameraParamsGPU computeCameraParams(int width, int height, int samples_per_pixel, int max_depth,
        glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 vup,
        double vfov, double defocus_angle, double focus_dist, int sphere_count) {

        CameraParamsGPU cam{};
        cam.image_width = width;
        cam.image_height = height;
        cam.samples_per_pixel = samples_per_pixel;
        cam.max_depth = max_depth;
        cam.defocus_angle = (float)defocus_angle;
        cam.sphere_count = sphere_count;

        cam.center = lookfrom;

        double theta = glm::radians(vfov);
        double h = tan(theta / 2);
        double viewport_height = 2 * h * focus_dist;
        double viewport_width = viewport_height * (double(width) / height);

        // Base orthonormee de la camera (u = right, v = up, w = back)
        glm::vec3 w = glm::normalize(lookfrom - lookat);
        glm::vec3 u = glm::normalize(glm::cross(vup, w));
        glm::vec3 v = glm::cross(w, u);

        cam.u = u;
        cam.v = v;
        cam.w = w;

        glm::vec3 viewport_u = (float)viewport_width * u;
        glm::vec3 viewport_v = (float)viewport_height * (-v);

        cam.pixel_delta_u = viewport_u / (float)width;
        cam.pixel_delta_v = viewport_v / (float)height;

        glm::vec3 viewport_upper_left = cam.center - (float)focus_dist * w - viewport_u / 2.0f - viewport_v / 2.0f;
        cam.pixel00_loc = viewport_upper_left + 0.5f * (cam.pixel_delta_u + cam.pixel_delta_v);

        double defocus_radius = focus_dist * tan(glm::radians(defocus_angle / 2.0));
        cam.defocus_disk_u = u * (float)defocus_radius;
        cam.defocus_disk_v = v * (float)defocus_radius;

        return cam;
    }


    void writePPM(const std::string& path, int width, int height) {
        void* data;
        VkDeviceSize size = sizeof(glm::vec4) * width * height;
        vkMapMemory(device, outputMemory, 0, size, 0, &data);

        glm::vec4* pixelsData = reinterpret_cast<glm::vec4*>(data);

        std::ofstream out(path);
        out << "P3\n" << width << ' ' << height << "\n255\n";

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                glm::vec4 c = pixelsData[j * width + i];
                int r = int(255.999 * glm::clamp(c.r, 0.0f, 1.0f));
                int g = int(255.999 * glm::clamp(c.g, 0.0f, 1.0f));
                int b = int(255.999 * glm::clamp(c.b, 0.0f, 1.0f));
                out << r << ' ' << g << ' ' << b << '\n';
            }
        }

        vkUnmapMemory(device, outputMemory);
        std::cout << "Image ecrite dans " << path << "\n";
    }

    void cleanup() {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, outputMemory, nullptr);
        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vkFreeMemory(device, uniformMemory, nullptr);
        vkDestroyBuffer(device, sphereBuffer, nullptr);
        vkFreeMemory(device, sphereMemory, nullptr);
        vkDestroyBuffer(device, materialBuffer, nullptr);
        vkFreeMemory(device, materialMemory, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};


int main() {
    try {
        std::vector<SphereGPU> spheres;
        std::vector<MaterialGPU> materials;

        // --- Materiau 0 : sol (lambertian gris-vert), comme le tuto ---
        materials.push_back({ glm::vec3(0.8f, 0.8f, 0.0f), 0.0f, 0.0f, /*type=*/0, 0,0 });
        spheres.push_back({ glm::vec3(0, -100.5f, -1), 100.0f, 0, 0,0,0 });

        // --- Materiau 1 : centre (lambertian bleute) ---
        materials.push_back({ glm::vec3(0.1f, 0.2f, 0.5f), 0.0f, 0.0f, /*type=*/0, 0,0 });
        spheres.push_back({ glm::vec3(0, 0, -1.2f), 0.5f, 1, 0,0,0 });

        // --- Materiau 2 : gauche (dielectric, verre) ---
        materials.push_back({ glm::vec3(1,1,1), 0.0f, 1.50f, /*type=*/2, 0,0 });
        spheres.push_back({ glm::vec3(-1.0f, 0, -1), 0.5f, 2, 0,0,0 });

        // --- Materiau 3 : bulle d'air dans le verre (indice inverse) ---
        materials.push_back({ glm::vec3(1,1,1), 0.0f, 1.0f / 1.50f, /*type=*/2, 0,0 });
        spheres.push_back({ glm::vec3(-1.0f, 0, -1), 0.4f, 3, 0,0,0 });

        // --- Materiau 4 : droite (metal dore, leger flou) ---
        materials.push_back({ glm::vec3(0.8f, 0.6f, 0.2f), 1.0f, 0.0f, /*type=*/1, 0,0 });
        spheres.push_back({ glm::vec3(1.0f, 0, -1), 0.5f, 4, 0,0,0 });

        // --- Camera positionnable + defocus blur, comme en fin de tuto ---
        glm::vec3 lookfrom(-2, 2, 1);
        glm::vec3 lookat(0, 0, -1);
        glm::vec3 vup(0, 1, 0);
        double vfov = 20.0;
        double defocus_angle = 0.6;
        double focus_dist = 3.4;

        VulkanCompute app;
        app.run(/*width=*/400, /*height=*/225, /*samples_per_pixel=*/100, /*max_depth=*/50,
            lookfrom, lookat, vup, vfov, defocus_angle, focus_dist,
            spheres, materials);
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }
    return 0;
}

