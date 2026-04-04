use nelfie::app::context::NelfieContext;

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    env_logger::try_init_from_env(env_logger::Env::default().default_filter_or("debug"))
        .unwrap_or(());
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");


    // コンテキスト初期化
    let ob_ctx = NelfieContext::new().await;
    if let Err(e) = ob_ctx.initialize_before_bot_start().await {
        eprintln!("failed to initialize voicevox before bot startup: {}", e);
        return;
    }

    if let Err(e) = ob_ctx.start_discord().await {
        eprintln!("failed to start discord bot: {}", e);
        return;
    }

    println!("nelfie started. Press Ctrl-C to shutdown...");
    if let Err(e) = tokio::signal::ctrl_c().await {
        eprintln!("failed to listen for Ctrl-C: {}", e);
    }
    println!("received Ctrl-C, shutting down...");

    if let Err(e) = ob_ctx.shutdown().await {
        eprintln!("engine shutdown error: {}", e);
    }
    println!("shutdown complete. Exiting.");
}
