//! Voxelicous Engine Benchmarks

use tracing::info;
#[cfg(feature = "profiling-tracy")]
use tracing_subscriber::EnvFilter;
#[cfg(feature = "profiling-tracy")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    #[cfg(feature = "profiling-tracy")]
    {
        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,voxelicous_app=trace,voxelicous_world=trace,voxelicous_render=trace,voxelicous_gpu=trace,voxelicous_benchmark=trace",
            )
        });
        let tracy_layer = tracing_tracy::TracyLayer::default();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .with(tracy_layer)
            .init();
    }
    #[cfg(not(feature = "profiling-tracy"))]
    {
        tracing_subscriber::fmt::init();
    }
    info!("Voxelicous Engine Benchmarks");
    info!("Run with: cargo bench -p voxelicous-voxel");
}
