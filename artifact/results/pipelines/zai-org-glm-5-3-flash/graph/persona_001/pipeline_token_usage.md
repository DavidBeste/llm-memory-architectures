# Pipeline Token-Usage Ledger

- Incremental exact tokens: `700,796.0`
- Logical-workload exact tokens: `700,796.0`
- Complete telemetry: `False`

Unavailable or partially covered components are not treated as zero.

| Component | Stage | Modality | Operation | Provider | Model | Availability | Calls | Coverage | Input | Cached input | Output | Reasoning | Total | Reused |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| graphiti_ingestion_llm | memory_initialization | generation | extract_entities_relationships_and_summaries | graphiti | zai-org/GLM-5.3-Flash | exact | 88/? | n/a | 122,351.0 | n/a | 18,529.0 | n/a | 140,880.0 | no |
| graphiti_ingestion_embeddings | memory_initialization | embedding | embed_ingested_graph_objects | graphiti | text-embedding-3-large | exact | 142/? | n/a | 4,017.0 | n/a | n/a | n/a | 4,017.0 | no |
| graphiti_retrieval_embeddings | online_memory_preparation | embedding | embed_graph_search_queries | graphiti | text-embedding-3-large | exact | 49/? | n/a | 2,362.0 | n/a | n/a | n/a | 2,362.0 | no |
| graphiti_internal_reranker | online_memory_preparation | reranking | graphiti_search_recipe_reranking | graphiti | zai-org/GLM-5.3-Flash | unavailable | 0/? | n/a | n/a | n/a | n/a | n/a | n/a | no |
| response_generation | online | generation | generate_recipient_response | letta | zai-org/GLM-5.3-Flash | exact | 49/49 | 100% | 186,076.0 | n/a | 93,270.0 | n/a | 279,346.0 | no |
| explicit_memory_reranking | online_memory_preparation | reranking | rerank_memory_candidates | None | zai-org/GLM-5.3-Flash | exact | 49/49 | 100% | 109,732.0 | n/a | 15,191.0 | n/a | 124,923.0 | no |
| exposure_judging | evaluation | judging | identify_exposed_attributes | responses | gpt-5.2 | exact | 49/49 | 100% | 131,139.0 | n/a | 18,129.0 | n/a | 149,268.0 | no |
| context_0.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_1.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_2.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_3.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_4.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_5.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_6.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_7.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_8.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_9.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_10.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_11.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_12.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_13.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_14.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_15.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_16.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_17.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_18.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_19.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_20.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_21.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_22.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_23.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_24.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_25.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_26.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_27.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_28.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_29.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_30.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_31.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_32.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_33.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_34.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_35.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_36.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_37.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_38.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_39.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_40.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_41.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_42.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_43.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_44.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_45.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_46.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_47.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| context_48.context_labeling | evaluation_setup | labeling | sample_context_privacy_labels | None | qwen3.8-27b | unavailable | 0/30 | 0% | n/a | n/a | n/a | n/a | n/a | no |

additive_exact_total sums only exact fully covered components; unavailable and partial components are never treated as zero; incremental_exact_total excludes reused artifacts, while logical_exact_total represents the complete workload using preserved usage
