# Pipeline Token-Usage Ledger

- Incremental exact tokens: `3,308,493.0`
- Logical-workload exact tokens: `3,308,493.0`
- Complete telemetry: `False`

Unavailable or partially covered components are not treated as zero.

| Component | Stage | Modality | Operation | Provider | Model | Availability | Calls | Coverage | Input | Cached input | Output | Reasoning | Total | Reused |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| memobase_initialization_aggregate | memory_initialization | backend_internal | insert_flush_extract_and_index | memobase | None | unavailable | 0/? | n/a | n/a | n/a | n/a | n/a | n/a | no |
| memobase_profile_extraction_llm | memory_initialization | generation | flush_and_extract_profile | memobase | None | unavailable | 0/? | n/a | n/a | n/a | n/a | n/a | n/a | no |
| memobase_ingestion_embeddings | memory_initialization | embedding | embed_profile_or_event_memory | memobase | None | unavailable | 0/? | n/a | n/a | n/a | n/a | n/a | n/a | no |
| memobase_retrieval_embeddings | online_memory_preparation | embedding | embed_context_search_queries | memobase | None | unavailable | 0/49 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| memobase_context_retrieval | online_memory_preparation | backend_internal | build_profile_context | memobase | None | unavailable | 0/49 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| response_generation | online | generation | generate_recipient_response | letta | gpt-5.6-sol | exact | 490/490 | 100% | 1,762,589.0 | n/a | 112,806.0 | n/a | 1,875,395.0 | no |
| explicit_memory_reranking | online_memory_preparation | reranking | rerank_memory_candidates | None | gpt-5.6-sol | exact | 49/49 | 100% | 78,791.0 | n/a | 19,259.0 | n/a | 98,050.0 | no |
| exposure_judging | evaluation | judging | identify_exposed_attributes | responses | gpt-5.2 | exact | 490/490 | 100% | 1,262,711.0 | n/a | 72,337.0 | n/a | 1,335,048.0 | no |
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
