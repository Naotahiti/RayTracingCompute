// main.cpp — Ray Tracing in One Weekend, version "compute" avec Vulkan
//
// Prérequis projet (Visual Studio) :
//   - Vulkan SDK : include dirs + lib dirs + vulkan-1.lib en lien
//   - GLM        : header-only, juste ajouter le dossier include
//   - GLFW       : non utilisé ici (pas de fenêtre, calcul "offline")
//
// AVANT DE COMPILER CE .cpp :
//   Compiler le shader avec glslc (fourni par le Vulkan SDK) :
//     glslc raytrace.comp -o raytrace.spv
//   Placer raytrace.spv dans le même dossier que l'exécutable.

#include "vulkan/vulkan.h"
#include "glm/glm.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// Structure envoyée au shader (doit matcher EXACTEMENT le layout std430/std140
// de "CameraParams" dans raytrace.comp, padding inclus)
// ---------------------------------------------------------------------------
struct CameraParamsGPU {
    glm::vec3 center;          float _pad0;
    glm::vec3 pixel00_loc;     float _pad1;
    glm::vec3 pixel_delta_u;   float _pad2;
    glm::vec3 pixel_delta_v;   float _pad3;
    int       image_width = 400;
    int       image_height = 400;
    int       samples_per_pixel = 10;
    int       _pad4;
};

// ---------------------------------------------------------------------------
// Petite classe utilitaire pour ne pas répéter le code Vulkan partout
// ---------------------------------------------------------------------------
class VulkanCompute {
public:
    void run(int image_width, int image_height, int samples_per_pixel) {
        this->image_width = image_width;
        this->image_height = image_height;

        createInstance();
        pickPhysicalDevice();
        createLogicalDeviceAndQueue();
        createCommandPool();

        // --- Caméra (équivalent de camera::initialize()) ---
        CameraParamsGPU cam = computeCameraParams(image_width, image_height, samples_per_pixel);

        VkDeviceSize outputSize = sizeof(glm::vec4) * image_width * image_height;
        VkDeviceSize uniformSize = sizeof(CameraParamsGPU);

        createBuffer(outputSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            outputBuffer, outputMemory);

        createBuffer(uniformSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffer, uniformMemory);

        // Copie des paramètres caméra dans le buffer GPU
        void* data;
        vkMapMemory(device, uniformMemory, 0, uniformSize, 0, &data);
        memcpy(data, &cam, (size_t)uniformSize);
        vkUnmapMemory(device, uniformMemory);

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

    VkBuffer outputBuffer{}, uniformBuffer{};
    VkDeviceMemory outputMemory{}, uniformMemory{};

    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline computePipeline{};
    VkDescriptorPool descriptorPool{};
    VkDescriptorSet descriptorSet{};

    int image_width = 0, image_height = 0;

    // -----------------------------------------------------------------------
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
        physicalDevice = devices[0]; // simplification : on prend le premier GPU trouvé

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

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[2]{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
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
        poolSizes[0].descriptorCount = 1;
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

        VkWriteDescriptorSet writes[2]{};
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

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
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

        // local_size_x/y = 8 dans le shader -> on arrondit au multiple de 8 superieur
        uint32_t groupsX = (width + 7) / 8;
        uint32_t groupsY = (height + 7) / 8;
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue); // simple : on attend la fin avant de continuer
    }

    CameraParamsGPU computeCameraParams(int width, int height, int samples_per_pixel) {
        CameraParamsGPU cam{};
        cam.image_width = width;
        cam.image_height = height;
        cam.samples_per_pixel = samples_per_pixel;

        cam.center = glm::vec3(0, 0, 0);

        double focal_length = 1.0;
        double vfov = 90.0;
        double theta = glm::radians(vfov);
        double h = tan(theta / 2);
        double viewport_height = 2 * h * focal_length;
        double viewport_width = viewport_height * (double(width) / height);

        glm::vec3 viewport_u(viewport_width, 0, 0);
        glm::vec3 viewport_v(0, -viewport_height, 0);

        cam.pixel_delta_u = viewport_u / float(width);
        cam.pixel_delta_v = viewport_v / float(height);

        glm::vec3 viewport_upper_left = cam.center - glm::vec3(0, 0, (float)focal_length)
            - viewport_u / 2.0f - viewport_v / 2.0f;

        cam.pixel00_loc = viewport_upper_left + 0.5f * (cam.pixel_delta_u + cam.pixel_delta_v);

        return cam;
    }

    void writePPM(const std::string& path, int width, int height) {
        void* data;
        VkDeviceSize size = sizeof(glm::vec4) * width * height;
        vkMapMemory(device, outputMemory, 0, size, 0, &data);

        glm::vec4* pixels = reinterpret_cast<glm::vec4*>(data);

        std::ofstream out(path);
        out << "P3\n" << width << ' ' << height << "\n255\n";

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                glm::vec4 c = pixels[j * width + i];
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
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};

int main() {
    try {
        VulkanCompute app;
        app.run(/*width=*/400, /*height=*/225, /*samples_per_pixel=*/100);
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }
    return 0;
}
