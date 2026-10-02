# Pipeline Token-Usage Ledger

- Incremental exact tokens: `2,705,344.0`
- Logical-workload exact tokens: `2,705,344.0`
- Complete telemetry: `False`

Unavailable or partially covered components are not treated as zero.

| Component | Stage | Modality | Operation | Provider | Model | Availability | Calls | Coverage | Input | Cached input | Output | Reasoning | Total | Reused |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| graphiti_ingestion_llm | memory_initialization | generation | extract_entities_relationships_and_summaries | graphiti | gpt-5.6-sol | exact | 109/? | n/a | 159,969.0 | n/a | 23,544.0 | n/a | 183,513.0 | no |
| graphiti_ingestion_embeddings | memory_initialization | embedding | embed_ingested_graph_objects | graphiti | text-embedding-3-large | exact | 187/? | n/a | 4,973.0 | n/a | n/a | n/a | 4,973.0 | no |
| graphiti_retrieval_embeddings | online_memory_preparation | embedding | embed_graph_search_queries | graphiti | text-embedding-3-large | exact | 49/? | n/a | 2,362.0 | n/a | n/a | n/a | 2,362.0 | no |
| graphiti_internal_reranker | online_memory_preparation | reranking | graphiti_search_recipe_reranking | graphiti | gpt-5.6-sol | unavailable | 0/? | n/a | n/a | n/a | n/a | n/a | n/a | no |
| response_generation | online | generation | generate_recipient_response | letta | gpt-5.6-sol | exact | 490/490 | 100% | 881,683.0 | n/a | 122,973.0 | n/a | 1,004,656.0 | no |
| explicit_memory_reranking | online_memory_preparation | reranking | rerank_memory_candidates | None | gpt-5.6-sol | exact | 49/49 | 100% | 122,426.0 | n/a | 29,179.0 | n/a | 151,605.0 | no |
| exposure_judging | evaluation | judging | identify_exposed_attributes | responses | gpt-5.2 | exact | 490/490 | 100% | 1,264,354.0 | n/a | 93,881.0 | n/a | 1,358,235.0 | no |
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
