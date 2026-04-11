#[cfg(test)]
mod pipeline_tests {
    use crate::pipeline::{event_map, fan_out, select, Mapper, Transform};
    use crate::protocol::{CandidateConfig, EventMapRule};
    use crate::types::events::StreamingEvent;
    use futures::StreamExt;
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_selector_filtering() {
        let selector = select::Selector::new("choices.0.delta.content".to_string());

        // Input: Stream of simple frames simulating a chunked response
        let input_data = vec![
            json!({
                "choices": [{"delta": {"content": "Hello"}}]
            }),
            json!({
                "choices": [{"delta": {"content": " World"}}]
            }),
            json!({
                "choices": [] // Should be filtered out
            }),
            json!({"other": "ignored"}), // Should be filtered out
        ];

        let input_stream = futures::stream::iter(input_data).map(Ok);
        let output_stream = selector.transform(Box::pin(input_stream)).await.unwrap();

        let results: Vec<_> = output_stream.map(|r| r.unwrap()).collect().await;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], json!("Hello"));
        assert_eq!(results[1], json!(" World"));
    }

    #[tokio::test]
    async fn test_fan_out() {
        let config = CandidateConfig {
            fan_out: Some(true),
            candidate_id_path: None,
        };
        let fan_out = fan_out::FanOut::new(config);

        // Input: Stream with arrays
        let input_data = vec![
            json!(["Candidate A"]),
            json!(["Candidate B", "Candidate C"]),
        ];

        let input_stream = futures::stream::iter(input_data).map(Ok);
        let output_stream = fan_out.transform(Box::pin(input_stream)).await.unwrap();

        let results: Vec<_> = output_stream.map(|r| r.unwrap()).collect().await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], json!("Candidate A"));
        assert_eq!(results[1], json!("Candidate B"));
        assert_eq!(results[2], json!("Candidate C"));
    }

    fn openai_style_event_rules() -> Vec<EventMapRule> {
        vec![
            EventMapRule {
                match_expr: "exists($.choices[*].delta.content)".to_string(),
                emit: "PartialContentDelta".to_string(),
                fields: Some({
                    let mut m = HashMap::new();
                    m.insert("content".to_string(), "$.choices[*].delta.content".to_string());
                    m
                }),
            },
            EventMapRule {
                match_expr: "exists($.usage)".to_string(),
                emit: "Metadata".to_string(),
                fields: Some({
                    let mut m = HashMap::new();
                    m.insert("usage".to_string(), "$.usage".to_string());
                    m
                }),
            },
            EventMapRule {
                match_expr: "$.choices[*].finish_reason != null".to_string(),
                emit: "StreamEnd".to_string(),
                fields: Some({
                    let mut m = HashMap::new();
                    m.insert("finish_reason".to_string(), "$.choices[*].finish_reason".to_string());
                    m
                }),
            },
        ]
    }

    #[tokio::test]
    async fn test_rule_event_mapper_deepseek_stream() {
        let rules = openai_style_event_rules();
        let mapper = event_map::RuleBasedEventMapper::new(&rules).unwrap();

        let frames = vec![
            json!({"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
        ];

        let input = futures::stream::iter(frames).map(Ok);
        let events: Vec<StreamingEvent> = mapper
            .map(Box::pin(input))
            .await
            .unwrap()
            .filter_map(|r| async { r.ok() })
            .collect()
            .await;

        let content_deltas: Vec<&str> = events.iter().filter_map(|e| match e {
            StreamingEvent::PartialContentDelta { content, .. } => Some(content.as_str()),
            _ => None,
        }).collect();
        assert_eq!(content_deltas, vec!["Hello", " world"]);

        let has_stream_end = events.iter().any(|e| matches!(
            e,
            StreamingEvent::StreamEnd { finish_reason: Some(r) } if r == "stop"
        ));
        assert!(has_stream_end, "expected StreamEnd with finish_reason='stop', events: {:?}", events);
    }

    #[tokio::test]
    async fn test_rule_event_mapper_null_content_filtered() {
        let rules = openai_style_event_rules();
        let mapper = event_map::RuleBasedEventMapper::new(&rules).unwrap();

        let frames = vec![
            json!({"choices":[{"index":0,"delta":{"content":null},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":null},"finish_reason":"stop"}]}),
        ];

        let input = futures::stream::iter(frames).map(Ok);
        let events: Vec<StreamingEvent> = mapper
            .map(Box::pin(input))
            .await
            .unwrap()
            .filter_map(|r| async { r.ok() })
            .collect()
            .await;

        let content_deltas: Vec<&str> = events.iter().filter_map(|e| match e {
            StreamingEvent::PartialContentDelta { content, .. } => Some(content.as_str()),
            _ => None,
        }).collect();
        assert_eq!(content_deltas, vec!["Hi"], "null content should not produce PartialContentDelta");

        let has_stream_end = events.iter().any(|e| matches!(
            e,
            StreamingEvent::StreamEnd { finish_reason: Some(r) } if r == "stop"
        ));
        assert!(has_stream_end, "StreamEnd should be emitted for finish_reason='stop'");
    }

    #[tokio::test]
    async fn test_rule_event_mapper_final_candidate_handled() {
        let rules = vec![
            EventMapRule {
                match_expr: "exists($.choices[*].delta.content)".to_string(),
                emit: "PartialContentDelta".to_string(),
                fields: Some({
                    let mut m = HashMap::new();
                    m.insert("content".to_string(), "$.choices[*].delta.content".to_string());
                    m
                }),
            },
            EventMapRule {
                match_expr: "exists($.choices[*].finish_reason)".to_string(),
                emit: "FinalCandidate".to_string(),
                fields: Some({
                    let mut m = HashMap::new();
                    m.insert("finish_reason".to_string(), "$.choices[*].finish_reason".to_string());
                    m
                }),
            },
        ];
        let mapper = event_map::RuleBasedEventMapper::new(&rules).unwrap();

        let frames = vec![
            json!({"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
        ];

        let input = futures::stream::iter(frames).map(Ok);
        let events: Vec<StreamingEvent> = mapper
            .map(Box::pin(input))
            .await
            .unwrap()
            .filter_map(|r| async { r.ok() })
            .collect()
            .await;

        assert!(events.iter().any(|e| matches!(e, StreamingEvent::PartialContentDelta { .. })));
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamingEvent::StreamEnd { finish_reason: Some(r) } if r == "stop"
            )),
            "FinalCandidate emit should produce StreamEnd, events: {:?}", events
        );
    }
}
