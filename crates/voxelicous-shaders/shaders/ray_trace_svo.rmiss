#version 460
#extension GL_EXT_ray_tracing : require

// Ray payload - color result
layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    // Sky gradient based on ray direction
    vec3 ray_dir = gl_WorldRayDirectionEXT;
    float t = 0.5 * (ray_dir.y + 1.0);

    vec3 sky_color = mix(
        vec3(0.8, 0.85, 0.9),   // Horizon color (light gray-blue)
        vec3(0.4, 0.6, 0.9),    // Zenith color (blue)
        t
    );

    payload = vec4(sky_color, 1.0);
}
