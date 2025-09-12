#ifndef CLIP_TOKENIZER_HPP
#define CLIP_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

class CLIPTokenizer {
public:
    CLIPTokenizer();
    ~CLIPTokenizer();
    
    // Main tokenization function that matches Python clip.tokenize()
    std::vector<std::vector<int32_t>> tokenize(const std::string& text, int context_length = 77);
    
    // Encode text to token IDs (without padding/special tokens)
    std::vector<int32_t> encode(const std::string& text);
    
    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& tokens);
    
    // Get vocabulary size
    size_t vocab_size() const;
    
    // Special token IDs
    int32_t start_token_id() const { return 49406; }
    int32_t end_token_id() const { return 49407; }
    int32_t pad_token_id() const { return 0; }

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};


#endif // CLIP_TOKENIZER_HPP