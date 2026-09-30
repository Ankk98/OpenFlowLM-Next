/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.hpp
 * \brief RestHandler class and related declarations
 * \author OpenFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#pragma once

#include "AutoModel/all_models.hpp"
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "whisper/modeling_whisper.hpp"
#include "AutoDecisionModel/all_decision_model.hpp"
#include "AutoEmbeddingModel/all_embedding_model.hpp"
#endif
#include "model_list.hpp"
#include "program_args.hpp"


#include "model_downloader.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <functional>
#include "prompt_cache.hpp"

using json = nlohmann::ordered_json;

// Forward declaration
struct CancellationToken;

///@brief Stream callback type for sending streaming responses
using StreamResponseCallback = std::function<void(const json&, bool)>; // data, is_final

#include "server/openai_compat.hpp"
class RestHandler {
public:
    RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args);
    ~RestHandler();

    void handle_show(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response);

    void handle_generate(const json& request, 
                        std::function<void(const json&)> send_response,
                        StreamResponseCallback send_streaming_response,
                        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

    void handle_chat(const json& request,
                    std::function<void(const json&)> send_response, 
                    StreamResponseCallback send_streaming_response,
                    std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    

    void handle_embeddings(const json& request,
                          std::function<void(const json&)> send_response,
                          StreamResponseCallback send_streaming_response);
    

    /// `POST /v1/systemone`. One forward pass, one distribution per question,
    /// no generation and no streaming -- so the third callback exists only to
    /// match the shape of the other handlers, and a request that arrives with
    /// `"stream": true` is refused by name rather than silently answered
    /// unstreamed (a client that asked for SSE and received one JSON object
    /// waits for more events that will never come).
    void handle_systemone(const json& request,
                          std::function<void(const json&)> send_response);

    void handle_models(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_models_openai(const json& request,
                            std::function<void(const json&)> send_response,
                            StreamResponseCallback send_streaming_response);

    void handle_ps(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_version(const json& request,
                       std::function<void(const json&)> send_response,
                       StreamResponseCallback send_streaming_response);
    
    // Placeholder handlers for unimplemented endpoints
    void handle_pull(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_push(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_delete(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_copy(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_create(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);

    void handle_openai_chat_completion(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_audio_transcriptions(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_completion(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response,
        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

private:
    using ModelLoad = openai_compat::ModelLoad;
    /// \param model_field_present the request carried a "model" key. Without it an
    ///        omitted field and an explicit "" are the same string -- see
    ///        openai_compat::preflight().
    ModelLoad ensure_model_loaded(const std::string& model_tag, bool model_field_present = false);
    void ensure_asr_model_loaded(const std::string& model_tag);
    void ensure_embed_model_loaded(const std::string& model_tag);
    void configure_chat_engine_parameters(const json& options, const json& request);
    json build_nstream_response(std::string response_text,
                                stop_reason_t stop_reason = EOT_DETECTED);


    std::unique_ptr<AutoModel> auto_chat_engine;
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::unique_ptr<Whisper> whisper_engine;
    std::unique_ptr<AutoEmbeddingModel> auto_embedding_engine;
    /// A decision model, and it is a SEPARATE member from the embedding engine
    /// rather than a variant of it. They are different kinds of model: one
    /// turns text into a vector, the other turns a state and typed questions
    /// into a distribution per question. A server may load both, and a single
    /// engine member holding "whichever was loaded last" is how
    /// /v1/embeddings starts answering with a decision model's nonsense or
    /// /v1/systemone answers with vectors.
    std::unique_ptr<AutoDecisionModel> auto_decision_engine;
    std::string decision_model_tag;
    /// The server's decision temperature, and whether it was set. Two members
    /// rather than one because 1.0 is both "asked for a neutral scale" and
    /// "asked for nothing", and the difference is whether the container's
    /// fitted per-type calibration survives.
    double decision_temperature = 1.0;
    bool decision_temperature_overridden = false;
#endif
    /// The parsed command line, kept by reference. The decision model's
    /// temperature and thread count are read from it at LOAD time, which is
    /// why a route cannot change them per request: they are server
    /// configuration, and a per-request temperature would be a field the pinned
    /// schema does not have.
    program_args_t& args;
    oflm_rt::device npu_device_inst;
    model_list& supported_models;
    ModelDownloader& downloader;
    std::string current_model_tag;
    std::string default_model_tag;
    bool modelscope;
    bool asr;
    std::string asr_model_tag;
    bool embed;
    // Which embedding model --embed loads, from --embeddingmodel.
    // Empty means embed-gemma:300m, so an existing command line keeps
    // its behaviour exactly.
    std::string embedding_model_tag;
    int prefill_chunk_len;
    int generate_context_id;
    int chat_context_id;
    int ctx_length;
    int img_pre_resize;
    std::string last_question;
    bool preemption;
    PromptCache prompt_cache;
};