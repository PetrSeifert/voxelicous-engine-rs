#version 460
#extension GL_EXT_ray_tracing : require

// Ray payload - color result
layout(location = 0) rayPayloadInEXT vec4 payload;

// Hit attributes from intersection shader
// xyz = surface normal, w = block_id (as float bits)
hitAttributeEXT vec4 hit_attribs;

// ============================================================================
// Shading Functions
// ============================================================================

vec3 get_block_color(uint block_id) {
    // Block colors (matching voxelicous-core BlockId constants)
    // AIR = 0, STONE = 1, DIRT = 2, GRASS = 3
    switch (block_id) {
        case 1u: return vec3(0.5, 0.5, 0.5);         // Stone - gray
        case 2u: return vec3(0.54, 0.35, 0.17);      // Dirt - brown
        case 3u: return vec3(0.34, 0.49, 0.27);      // Grass - green
        default: return vec3(0.8, 0.2, 0.8);         // Unknown - magenta
    }
}

// ============================================================================
// Closest-Hit Shader Main
// ============================================================================

void main() {
    // Unpack hit attributes
    vec3 normal = hit_attribs.xyz;
    uint block_id = floatBitsToUint(hit_attribs.w);

    vec3 base_color = get_block_color(block_id);

    // Simple directional light (sun)
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    float ndotl = max(dot(normal, light_dir), 0.0);

    // Ambient + diffuse lighting
    float ambient = 0.3;
    float diffuse = 0.7 * ndotl;

    vec3 final_color = base_color * (ambient + diffuse);

    payload = vec4(final_color, 1.0);
}
