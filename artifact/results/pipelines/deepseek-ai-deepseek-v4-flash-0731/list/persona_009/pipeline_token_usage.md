# Pipeline Token-Usage Ledger

- Incremental exact tokens: `342,603.0`
- Logical-workload exact tokens: `342,603.0`
- Complete telemetry: `False`

Unavailable or partially covered components are not treated as zero.

| Component | Stage | Modality | Operation | Provider | Model | Availability | Calls | Coverage | Input | Cached input | Output | Reasoning | Total | Reused |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| letta_archival_ingestion_embeddings | memory_initialization | embedding | embed_archival_passages | letta | text-embedding-3-large | unavailable | 150/149 | 100% | n/a | n/a | n/a | n/a | n/a | no |
| letta_archival_search_embeddings | online_memory_preparation | embedding | embed_archival_search_queries | letta | text-embedding-3-large | unavailable | 0/49 | 0% | n/a | n/a | n/a | n/a | n/a | no |
| response_generation | online | generation | generate_recipient_response | letta | deepseek-ai/DeepSeek-V4-Flash-0731 | exact | 49/49 | 100% | 108,697.0 | n/a | 19,191.0 | n/a | 127,888.0 | no |
| explicit_memory_reranking | online_memory_preparation | reranking | rerank_memory_candidates | None | deepseek-ai/DeepSeek-V4-Flash-0731 | exact | 49/49 | 100% | 81,889.0 | n/a | 2,412.0 | n/a | 84,301.0 | no |
| exposure_judging | evaluation | judging | identify_exposed_attributes | responses | gpt-5.2 | exact | 49/49 | 100% | 112,754.0 | n/a | 17,660.0 | n/a | 130,414.0 | no |
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
