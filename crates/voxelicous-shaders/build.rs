//! Build script to compile GLSL shaders to SPIR-V.

use shaderc::{Compiler, ShaderKind};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new("shaders");

    // Rerun if shaders change
    println!("cargo:rerun-if-changed=shaders/");

    // Create compiler
    let compiler = Compiler::new().expect("Failed to create shader compiler");

    // Compile ray_march_svo.comp (compute fallback)
    compile_shader(
        &compiler,
        shader_dir.join("ray_march_svo.comp"),
        Path::new(&out_dir).join("ray_march_svo.spv"),
        ShaderKind::Compute,
    );

    // Compile ray tracing shaders (feature-gated)
    #[cfg(feature = "ray_tracing")]
    {
        compile_shader(
            &compiler,
            shader_dir.join("ray_trace_svo.rgen"),
            Path::new(&out_dir).join("ray_trace_svo_rgen.spv"),
            ShaderKind::RayGeneration,
        );

        compile_shader(
            &compiler,
            shader_dir.join("ray_trace_svo.rmiss"),
            Path::new(&out_dir).join("ray_trace_svo_rmiss.spv"),
            ShaderKind::Miss,
        );

        compile_shader(
            &compiler,
            shader_dir.join("ray_trace_svo.rint"),
            Path::new(&out_dir).join("ray_trace_svo_rint.spv"),
            ShaderKind::Intersection,
        );

        compile_shader(
            &compiler,
            shader_dir.join("ray_trace_svo.rchit"),
            Path::new(&out_dir).join("ray_trace_svo_rchit.spv"),
            ShaderKind::ClosestHit,
        );
    }
}

fn compile_shader(
    compiler: &Compiler,
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    kind: ShaderKind,
) {
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    let source = fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", input_path, e));

    let file_name = input_path.file_name().unwrap().to_str().unwrap();

    let mut options = shaderc::CompileOptions::new().expect("Failed to create compile options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let result = compiler
        .compile_into_spirv(&source, kind, file_name, "main", Some(&options))
        .unwrap_or_else(|e| panic!("Failed to compile shader {:?}: {}", input_path, e));

    if result.get_num_warnings() > 0 {
        println!("cargo:warning=Shader warnings in {:?}:", input_path);
        // Warnings are printed by shaderc
    }

    fs::write(
        output_path,
        bytemuck::cast_slice::<u32, u8>(result.as_binary()),
    )
    .unwrap_or_else(|e| panic!("Failed to write shader {:?}: {}", output_path, e));

    println!("Compiled {:?} -> {:?}", input_path, output_path);
}
