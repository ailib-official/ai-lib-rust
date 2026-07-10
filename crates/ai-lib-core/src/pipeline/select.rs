use crate::pipeline::{PipelineError, Transform};
use crate::utils::PathMapper;
use crate::{BoxStream, PipeResult};
use futures::StreamExt;
use serde_json::Value;

/// Selector filters the stream to include only relevant frames
/// typically matching a specific JSON path (e.g., "choices.0.delta")
pub struct Selector {
    path: String,
    evaluator: crate::utils::json_path::JsonPathEvaluator,
}

impl Selector {
    pub fn try_new(path: String) -> Result<Self, PipelineError> {
        let evaluator = crate::utils::json_path::JsonPathEvaluator::new(&path)
            .or_else(|_| {
                // Fallback for simple paths if evaluator creation fails
                crate::utils::json_path::JsonPathEvaluator::new(&format!("exists({})", path))
            })
            .map_err(|e| {
                PipelineError::Configuration(format!(
                    "Invalid frame_selector path '{}': {}",
                    path, e
                ))
            })?;
        Ok(Self { path, evaluator })
    }
}

#[async_trait::async_trait]
impl Transform for Selector {
    async fn transform(
        &self,
        input: BoxStream<'static, Value>,
    ) -> PipeResult<BoxStream<'static, Value>> {
        let path = self.path.clone();
        let evaluator = self.evaluator.clone();

        let stream = input.filter_map(move |result| {
            let path = path.clone();
            let evaluator = evaluator.clone();
            async move {
                match result {
                    Ok(value) => {
                        // Apply selection logic
                        // 1. If it's a condition, return whole frame if matches
                        if path.contains("exists(")
                            || path.contains("==")
                            || path.contains("||")
                            || path.contains("&&")
                        {
                            if evaluator.matches(&value) {
                                return Some(Ok(value));
                            } else {
                                return None;
                            }
                        }

                        // 2. Simple path selection
                        PathMapper::get_path(&value, &path).cloned().map(Ok)
                    }
                    Err(e) => Some(Err(e)), // Propagate errors
                }
            }
        });

        Ok(Box::pin(stream))
    }
}

pub fn create_selector(path: &str) -> Result<Box<dyn Transform>, PipelineError> {
    Ok(Box::new(Selector::try_new(path.to_string())?))
}
