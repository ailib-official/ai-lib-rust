use ai_lib_rust::AiClient;

/// Prefer CI / developer `AI_PROTOCOL_DIR`; do not overwrite with a machine-local path.
fn protocol_dir_for_tests() {
    if std::env::var_os("AI_PROTOCOL_DIR").is_some()
        || std::env::var_os("AI_PROTOCOL_PATH").is_some()
    {
        return;
    }
    // Local fallbacks used by ProtocolLoader when env is unset.
    for candidate in ["ai-protocol", "../ai-protocol", "../../ai-protocol"] {
        if std::path::Path::new(candidate).exists() {
            std::env::set_var("AI_PROTOCOL_DIR", candidate);
            return;
        }
    }
}

#[tokio::test]
async fn test_loading_all_providers() {
    protocol_dir_for_tests();

    let providers = vec!["openai", "anthropic", "gemini", "deepseek", "groq", "qwen"];

    for provider in providers {
        // Test direct provider loading via a model-id-like string
        let client = AiClient::new(&format!("{}/some-model", provider)).await;
        assert!(
            client.is_ok(),
            "Failed to load provider '{}': {:?}",
            provider,
            client.err()
        );
        let client = client.unwrap();
        assert_eq!(client.manifest.id, provider);
    }
}

#[tokio::test]
async fn test_loading_registered_models() {
    protocol_dir_for_tests();

    let models = vec![
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "deepseek/deepseek-chat",
        "gemini/gemini-1.5-pro",
        "groq/llama3-70b-8192",
        "qwen/qwen-max",
    ];

    for model in models {
        let client = AiClient::new(model).await;
        assert!(
            client.is_ok(),
            "Failed to load registered model '{}': {:?}",
            model,
            client.err()
        );
    }
}
