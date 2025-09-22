#include "mobileclip_tokenizer.hpp"
#include <regex>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cctype>
#include <limits>
#include <codecvt>
#include <locale>
#include <iostream>

// Hash function for pairs - must be in global namespace
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class MOBILECLIPTokenizer::Impl {
public:
    Impl();
    
    std::vector<int32_t> encode(const std::string& text);
    std::string decode(const std::vector<int32_t>& tokens);
    size_t vocab_size() const { return encoder.size(); }

private:
    std::unordered_map<std::string, int32_t> encoder;
    std::unordered_map<int32_t, std::string> decoder;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;
    std::unordered_map<uint8_t, std::string> byte_encoder;
    std::unordered_map<std::string, uint8_t> byte_decoder;
    std::unordered_map<std::string, std::string> cache;
    std::regex pat;
    
    void initialize_byte_encoder();
    void load_vocabulary();
    std::string bpe(const std::string& token);
    std::vector<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
    std::string basic_clean(const std::string& text);
    std::string whitespace_clean(const std::string& text);
    std::string lowercase(const std::string& text);
    std::vector<std::string> utf8_to_bytes(const std::string& text);
};

MOBILECLIPTokenizer::Impl::Impl() : 
    pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]+)", 
        std::regex::icase) {
    initialize_byte_encoder();
    load_vocabulary();
    cache["<|startoftext|>"] = "<|startoftext|>";
    cache["<|endoftext|>"] = "<|endoftext|>";
}

void MOBILECLIPTokenizer::Impl::initialize_byte_encoder() {
    std::vector<int> bs;
    
    // Add printable ASCII characters
    for (int i = 33; i <= 126; i++) bs.push_back(i);  // ! to ~
    for (int i = 161; i <= 172; i++) bs.push_back(i); // ¡ to ¬
    for (int i = 174; i <= 255; i++) bs.push_back(i); // ® to ÿ
    
    std::vector<int> cs = bs;
    int n = 0;
    
    // Add remaining bytes
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    
    // Create byte to unicode mappings
    for (size_t i = 0; i < bs.size(); i++) {
        if (cs[i] < 128) {
            byte_encoder[bs[i]] = std::string(1, static_cast<char>(cs[i]));
        } else {
            // For values >= 128, we need to encode as UTF-8
            std::string result;
            if (cs[i] < 0x800) {
                result += static_cast<char>(0xC0 | (cs[i] >> 6));
                result += static_cast<char>(0x80 | (cs[i] & 0x3F));
            } else {
                result += static_cast<char>(0xE0 | (cs[i] >> 12));
                result += static_cast<char>(0x80 | ((cs[i] >> 6) & 0x3F));
                result += static_cast<char>(0x80 | (cs[i] & 0x3F));
            }
            byte_encoder[bs[i]] = result;
        }
        byte_decoder[byte_encoder[bs[i]]] = bs[i];
    }
}

void MOBILECLIPTokenizer::Impl::load_vocabulary() {
    // Load BPE merges
    std::ifstream merges_file("../../data/bpe_merges.txt");
    if (!merges_file.is_open()) {
        throw std::runtime_error("Could not open bpe_merges.txt");
    }
    
    std::string line;
    int rank = 0;
    while (std::getline(merges_file, line)) {
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string first, second;
            if (iss >> first >> second) {
                bpe_ranks[{first, second}] = rank++;
            }
        }
    }
    merges_file.close();
    
    // Build vocabulary
    std::vector<std::string> vocab;
    
    // Add base vocabulary (bytes_to_unicode values) in the correct order
    // The order must match Python's bytes_to_unicode() function
    // First add printable ASCII, then specific ranges, then remaining bytes
    std::vector<int> byte_order;
    
    // Same order as Python: printable ASCII
    for (int i = 33; i <= 126; i++) byte_order.push_back(i);  // ! to ~
    for (int i = 161; i <= 172; i++) byte_order.push_back(i); // ¡ to ¬
    for (int i = 174; i <= 255; i++) byte_order.push_back(i); // ® to ÿ
    
    // Add remaining bytes in order
    for (int b = 0; b < 256; b++) {
        if (std::find(byte_order.begin(), byte_order.end(), b) == byte_order.end()) {
            byte_order.push_back(b);
        }
    }
    
    // Now build vocabulary in this specific order
    for (int b : byte_order) {
        vocab.push_back(byte_encoder[b]);
    }
    
    // Add vocabulary with </w> suffix
    size_t base_size = vocab.size();
    for (size_t i = 0; i < base_size; i++) {
        vocab.push_back(vocab[i] + "</w>");
    }
    
    // Add merged tokens
    std::vector<std::pair<std::pair<std::string, std::string>, int>> sorted_ranks;
    for (const auto& pair : bpe_ranks) {
        sorted_ranks.push_back(pair);
    }
    std::sort(sorted_ranks.begin(), sorted_ranks.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& pair : sorted_ranks) {
        vocab.push_back(pair.first.first + pair.first.second);
    }
    
    // Add special tokens
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");
    
    // Create encoder/decoder mappings
    for (size_t i = 0; i < vocab.size(); i++) {
        encoder[vocab[i]] = static_cast<int32_t>(i);
        decoder[static_cast<int32_t>(i)] = vocab[i];
    }
}

std::vector<std::pair<std::string, std::string>> MOBILECLIPTokenizer::Impl::get_pairs(const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    
    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.push_back({word[i], word[i + 1]});
    }
    return pairs;
}

std::string MOBILECLIPTokenizer::Impl::bpe(const std::string& token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }
    
    std::vector<std::string> word;
    for (size_t i = 0; i < token.length() - 1; i++) {
        word.push_back(std::string(1, token[i]));
    }
    if (!token.empty()) {
        word.push_back(std::string(1, token.back()) + "</w>");
    }
    
    auto pairs = get_pairs(word);
    
    if (pairs.empty()) {
        return token + "</w>";
    }
    
    while (true) {
        // Find the pair with minimum rank
        int min_rank = std::numeric_limits<int>::max();
        std::pair<std::string, std::string> bigram;
        bool found = false;
        
        for (const auto& pair : pairs) {
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                bigram = pair;
                found = true;
            }
        }
        
        if (!found) break;
        
        // Merge the bigram
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == bigram.first && word[i + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        
        word = new_word;
        if (word.size() == 1) break;
        
        pairs = get_pairs(word);
    }
    
    // Join word pieces with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); i++) {
        if (i > 0) result += " ";
        result += word[i];
    }
    
    cache[token] = result;
    return result;
}

std::string MOBILECLIPTokenizer::Impl::basic_clean(const std::string& text) {
    // Simplified version - in production, use proper text cleaning
    return text;
}

std::string MOBILECLIPTokenizer::Impl::whitespace_clean(const std::string& text) {
    std::string result;
    bool prev_space = true;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }
    
    // Trim
    size_t first = result.find_first_not_of(' ');
    size_t last = result.find_last_not_of(' ');
    if (first == std::string::npos) return "";
    return result.substr(first, last - first + 1);
}

std::string MOBILECLIPTokenizer::Impl::lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<int32_t> MOBILECLIPTokenizer::Impl::encode(const std::string& text) {
    std::vector<int32_t> bpe_tokens;
    
    // Clean and lowercase text
    std::string cleaned = lowercase(whitespace_clean(basic_clean(text)));
    
    // For simplicity, we'll use a basic regex matching approach
    // In production, use a proper regex library with Unicode support
    std::regex word_regex(R"(\S+)");
    auto words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    for (auto it = words_begin; it != words_end; ++it) {
        std::string token = it->str();
        
        // Convert to byte encoding
        std::string byte_token;
        for (unsigned char c : token) {
            byte_token += byte_encoder[c];
        }
        
        // Apply BPE
        std::string bpe_result = bpe(byte_token);
        
        // Split by spaces and encode
        std::istringstream iss(bpe_result);
        std::string bpe_token;
        while (iss >> bpe_token) {
            auto it = encoder.find(bpe_token);
            if (it != encoder.end()) {
                bpe_tokens.push_back(it->second);
            }
        }
    }
    
    return bpe_tokens;
}

std::string MOBILECLIPTokenizer::Impl::decode(const std::vector<int32_t>& tokens) {
    std::string text;
    for (int32_t token : tokens) {
        auto it = decoder.find(token);
        if (it != decoder.end()) {
            text += it->second;
        }
    }
    
    // Convert byte encoding back to UTF-8
    std::string result;
    for (const auto& ch : text) {
        auto it = byte_decoder.find(std::string(1, ch));
        if (it != byte_decoder.end()) {
            result += static_cast<char>(it->second);
        }
    }
    
    // Replace </w> with spaces
    size_t pos = 0;
    while ((pos = result.find("</w>", pos)) != std::string::npos) {
        result.replace(pos, 4, " ");
        pos += 1;
    }
    
    return result;
}

// CLIPTokenizer public interface implementation
MOBILECLIPTokenizer::MOBILECLIPTokenizer() : pImpl(std::make_unique<Impl>()) {}

MOBILECLIPTokenizer::~MOBILECLIPTokenizer() = default;

std::vector<std::vector<int32_t>> MOBILECLIPTokenizer::tokenize(const std::string& text, int context_length) {
    // Encode the text
    auto tokens = pImpl->encode(text);
    
    // Create result with start and end tokens
    std::vector<int32_t> result;
    result.push_back(start_token_id());
    result.insert(result.end(), tokens.begin(), tokens.end());
    result.push_back(end_token_id());
    
    // Truncate if necessary
    if (result.size() > static_cast<size_t>(context_length)) {
        result.resize(context_length);
        result.back() = end_token_id();
    }
    
    // Pad to context_length
    while (result.size() < static_cast<size_t>(context_length)) {
        result.push_back(pad_token_id());
    }
    
    // Return as batch of 1
    return {result};
}

std::vector<int32_t> MOBILECLIPTokenizer::encode(const std::string& text) {
    return pImpl->encode(text);
}

std::string MOBILECLIPTokenizer::decode(const std::vector<int32_t>& tokens) {
    return pImpl->decode(tokens);
}

size_t MOBILECLIPTokenizer::vocab_size() const {
    return pImpl->vocab_size();
}

