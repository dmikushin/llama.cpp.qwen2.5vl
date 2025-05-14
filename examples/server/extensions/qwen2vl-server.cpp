#include "utils.hpp"

#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "base64.hpp"

// Required for image processing
#include "clip.h"
#include "llava.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cinttypes>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

using json = nlohmann::ordered_json;

// Forward declarations 
struct server_context;
struct server_slot;
struct common_sampler;
struct result_timings;
struct llava_context;

// Image embedding constants
static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

// Vision-language processing functions adapted from qwen2vl-cli.cpp
static bool qwen2vl_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id, struct clip_image_size * image_size) {
    int n_embd  = llama_model_n_embd(llama_get_model(ctx_llama));
    const int patch_size = 14 * 2;
    const int ph = image_size->height / patch_size + (image_size->height % patch_size > 0);
    const int pw = image_size->width / patch_size + (image_size->width % patch_size > 0);
    auto img_tokens = image_embed->n_image_pos;
    // llama_pos mrope_pos[img_tokens * 4];
    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(img_tokens * 4);

    for (int y = 0; y < ph; y++)
    {
        for (int x = 0; x < pw; x++)
        {
            int i = y * pw + x;
            mrope_pos[i] = *st_pos_id;
            mrope_pos[i + img_tokens] = *st_pos_id + y;
            mrope_pos[i + img_tokens * 2] = *st_pos_id + x;
            mrope_pos[i + img_tokens * 3] = 0;
        }
    }
    *st_pos_id += std::max(pw, ph);

    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(img_tokens * 4);

    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = img_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        // llama_pos batch_mrope_pos[n_eval * 4];
        std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
        memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

        llama_batch batch = {
            int32_t(n_eval),                // n_tokens
            nullptr,                        // token
            (image_embed->embed+i*n_embd),  // embed
            batch_mrope_pos.data(),         // pos
            nullptr,                        // n_seq_id
            nullptr,                        // seq_id
            nullptr,                        // logits
        };

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
        processed += n_eval;
    }
    return true;
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id) {
    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        // Add mrope position ids
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = *st_pos_id + (j % batch.n_tokens);
        }
        batch.pos = pos.data();

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
        *st_pos_id += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos) {
    std::string str2 = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id);
}

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

// Extract base64 image from prompt and create an embedding
static struct llava_image_embed * process_image_from_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count);

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

// Main function to process a multimodal prompt with image
static void process_multimodal_prompt(
    struct llama_context * ctx_llama, 
    struct clip_ctx * ctx_clip,
    struct llava_image_embed * image_embed, 
    const std::string & prompt, 
    const int n_batch,
    const int n_threads,
    const bool verbose_prompt,
    std::function<void(const std::string&, bool)> token_callback) {
    
    int n_past = 0;
    int cur_pos_id = 0;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<|vision_start|>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<|vision_pad|>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (verbose_prompt) {
            auto tmp = common_tokenize(ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (verbose_prompt) {
            auto tmp = common_tokenize(ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
        user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (verbose_prompt) {
            auto tmp = common_tokenize(ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llama, system_prompt.c_str(), n_batch, &n_past, &cur_pos_id, true);
    if (image_embed != nullptr) {
        auto image_size = clip_get_load_image_size(ctx_clip);
        qwen2vl_eval_image_embed(ctx_llama, image_embed, n_batch, &n_past, &cur_pos_id, image_size);
    }
    eval_string(ctx_llama, user_prompt.c_str(), n_batch, &n_past, &cur_pos_id, false);

    // Setup for generation
    struct common_params_sampling sampling;
    sampling.temp = 0.7f;
    sampling.top_k = 40;
    sampling.top_p = 0.95f;
    sampling.seed = 0;
    
    struct common_sampler * sampler = common_sampler_init(llama_get_model(ctx_llama), sampling);
    if (!sampler) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return;
    }

    // Generate response
    const int max_tokens = 1024; // Default max tokens to generate
    bool stop_generation = false;

    for (int i = 0; i < max_tokens && !stop_generation; i++) {
        const llama_token id = common_sampler_sample(sampler, ctx_llama, -1);
        common_sampler_accept(sampler, id, true);

        const llama_model * model = llama_get_model(ctx_llama);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        std::string token_str;
        if (llama_vocab_is_eog(vocab, id)) {
            token_str = "</s>";
            stop_generation = true;
        } else {
            token_str = common_token_to_piece(ctx_llama, id);
        }

        token_callback(token_str, stop_generation);
        
        // Check for other stop conditions
        if (token_str == "<|im_end|>" || 
            token_str.find("###") != std::string::npos || 
            token_str.find("USER:") != std::string::npos) {
            stop_generation = true;
        }
        
        // Continue generating if not stopped
        if (!stop_generation) {
            eval_id(ctx_llama, id, &n_past, &cur_pos_id);
        }
    }

    common_sampler_free(sampler);
}

// Function to add Qwen2.5VL server routes to the main server
static void add_qwen2vl_routes(httplib::Server * svr, server_context * ctx_server) {
    if (!svr) {
        return;
    }
    
    svr->Post("/qwen2vl", [ctx_server](const httplib::Request & req, httplib::Response & res) {
        try {
            json request_data = json::parse(req.body);
            
            // Extract parameters
            std::string prompt = request_data["prompt"];
            bool stream = json_value(request_data, "stream", true);
            
            // Ensure the model is compatible
            if (!ctx_server->params_base.mmproj.path.empty() && prompt_contains_image(prompt)) {
                // Create a llava context for image processing
                struct llava_context ctx_llava;
                ctx_llava.ctx_llama = ctx_server->ctx;
                ctx_llava.model = ctx_server->model;
                
                // Load CLIP model
                ctx_llava.ctx_clip = clip_model_load(ctx_server->params_base.mmproj.path.c_str(), GGML_LOG_LEVEL_INFO);
                if (!ctx_llava.ctx_clip) {
                    json error = {
                        {"error", "Failed to load CLIP model"},
                        {"status", 500}
                    };
                    res.set_content(error.dump(), MIMETYPE_JSON);
                    res.status = 500;
                    return;
                }
                
                // Process image from base64
                auto image_embed = process_image_from_base64(ctx_llava.ctx_clip, ctx_server->params_base.cpuparams.n_threads, prompt);
                if (!image_embed) {
                    json error = {
                        {"error", "Failed to process image"},
                        {"status", 500}
                    };
                    res.set_content(error.dump(), MIMETYPE_JSON);
                    res.status = 500;
                    return;
                }
                
                // Remove image from prompt for text-only part
                std::string clean_prompt = remove_image_from_prompt(prompt);
                
                // For streaming responses
                if (stream) {
                    res.set_header("Content-Type", "text/event-stream");
                    res.set_header("Cache-Control", "no-cache");
                    res.set_chunked_content_provider("text/event-stream", [&ctx_llava, &image_embed, &prompt, &ctx_server](size_t, httplib::DataSink & sink) {
                        std::string full_response;
                        
                        // Process the multimodal prompt
                        process_multimodal_prompt(ctx_llava.ctx_llama, ctx_llava.ctx_clip, image_embed, prompt, 
                            ctx_server->params_base.n_batch, ctx_server->params_base.cpuparams.n_threads, 
                            ctx_server->params_base.verbosity > 1,
                            [&sink, &full_response](const std::string& token, bool is_final) {
                                full_response += token;
                                
                                json chunk = {
                                    {"token", token},
                                    {"generated_text", full_response},
                                    {"complete", is_final}
                                };
                                
                                std::string data = "data: " + chunk.dump() + "\n\n";
                                sink.write(data.c_str(), data.size());
                                
                                if (is_final) {
                                    sink.write("data: [DONE]\n\n", 14);
                                    return false; // End streaming
                                }
                                return true; // Continue streaming
                            });
                        
                        // Cleanup
                        llava_image_embed_free(image_embed);
                        clip_free(ctx_llava.ctx_clip);
                        
                        return true; // Streaming completed successfully
                    });
                    
                    return;
                } else {
                    // For non-streaming responses
                    std::string full_response;
                    
                    // Process the multimodal prompt
                    process_multimodal_prompt(ctx_llava.ctx_llama, ctx_llava.ctx_clip, image_embed, prompt, 
                        ctx_server->params_base.n_batch, ctx_server->params_base.cpuparams.n_threads, 
                        ctx_server->params_base.verbosity > 1,
                        [&full_response](const std::string& token, bool) {
                            full_response += token;
                            return true;
                        });
                    
                    // Create response JSON
                    json response = {
                        {"generated_text", full_response},
                        {"prompt", clean_prompt}
                    };
                    
                    res.set_content(response.dump(), MIMETYPE_JSON);
                    
                    // Cleanup
                    llava_image_embed_free(image_embed);
                    clip_free(ctx_llava.ctx_clip);
                    
                    return;
                }
            } else {
                json error = {
                    {"error", "No image found in prompt or mmproj model not loaded"},
                    {"status", 400}
                };
                res.set_content(error.dump(), MIMETYPE_JSON);
                res.status = 400;
                return;
            }
            
        } catch (const std::exception & e) {
            json error = {
                {"error", std::string("Exception: ") + e.what()},
                {"status", 500}
            };
            res.set_content(error.dump(), MIMETYPE_JSON);
            res.status = 500;
        }
    });
    
    // Add a health check endpoint to verify the Qwen2.5VL server is running
    svr->Get("/qwen2vl/health", [ctx_server](const httplib::Request &, httplib::Response & res) {
        json health = {
            {"status", "ok"},
            {"model", ctx_server->params_base.model.path},
            {"mmproj", ctx_server->params_base.mmproj.path}
        };
        res.set_content(health.dump(), MIMETYPE_JSON);
    });
}

// Export this function to be called from the main server to register the Qwen2.5VL routes
extern "C" void register_qwen2vl_routes(httplib::Server * svr, void * ctx) {
    add_qwen2vl_routes(svr, static_cast<server_context*>(ctx));
}