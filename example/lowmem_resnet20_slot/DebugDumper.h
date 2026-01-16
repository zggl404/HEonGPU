#ifndef LOWMEM_RESNET20_DEBUG_DUMPER_H
#define LOWMEM_RESNET20_DEBUG_DUMPER_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "LowMemAdapter.h"

class DebugDumper {
  public:
    explicit DebugDumper(lowmem::FHEController* controller)
        : controller_(controller)
    {
    }

    void set_enabled(bool enabled) { enabled_ = enabled; }

    void set_count(int count)
    {
        if (count > 0) {
            count_ = count;
        }
    }

    void set_layer_regex(const std::string& pattern)
    {
        layer_regex_ = pattern;
        regex_valid_ = true;
        if (!pattern.empty()) {
            try {
                compiled_regex_ = std::regex(pattern);
            } catch (const std::regex_error&) {
                regex_valid_ = false;
            }
        }
    }

    bool should_dump(const std::string& tag) const
    {
        if (!enabled_) {
            return false;
        }
        if (layer_regex_.empty()) {
            return true;
        }
        if (!regex_valid_) {
            return true;
        }
        return std::regex_search(tag, compiled_regex_);
    }

    void set_tolerances(double tol_abs, double tol_rel)
    {
        if (tol_abs > 0.0) {
            tol_abs_ = tol_abs;
        }
        if (tol_rel > 0.0) {
            tol_rel_ = tol_rel;
        }
    }

    bool load_reference_log(const std::string& path)
    {
        if (path.empty()) {
            return false;
        }
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        std::regex line_re(R"(\[DBG\]\s+(.+?):\s*\[(.*)\])");
        std::string line;
        size_t count = 0;
        while (std::getline(in, line)) {
            std::smatch match;
            if (!std::regex_search(line, match, line_re)) {
                continue;
            }
            const std::string tag = match[1].str();
            const std::string list = match[2].str();
            std::vector<double> values = parse_list(list);
            if (!values.empty()) {
                ref_[tag] = values;
                ++count;
            }
        }
        ref_loaded_ = !ref_.empty();
        return ref_loaded_;
    }

    void finalize(const std::string& json_path)
    {
        if (!ref_loaded_) {
            return;
        }
        int bad_count = 0;
        for (const auto& entry : diff_entries_) {
            if (entry.max_abs > tol_abs_ || entry.max_rel > tol_rel_) {
                ++bad_count;
            }
        }
        write_json(json_path);
    }

    void dump_ct(const std::string& tag, const lowmem::Ctxt& ct,
                 int count = -1)
    {
        if (!should_dump(tag) || controller_ == nullptr) {
            return;
        }
        const int use_count = (count > 0) ? count : count_;
        std::vector<double> vals = controller_->decrypt_tovector(ct, use_count);
        print_values(tag, vals, use_count);
        print_meta(tag, ct.level(), ct.scale());
        if (ref_loaded_) {
            compare_and_report(tag, vals, ct.level(), ct.scale(), use_count);
        }
    }

    void dump_pt(const std::string& tag, const lowmem::Ptxt& pt,
                 int count = -1)
    {
        if (!should_dump(tag) || controller_ == nullptr) {
            return;
        }
        const int use_count = (count > 0) ? count : count_;
        std::vector<double> vals =
            controller_->decode_plaintext(pt, use_count);
        print_values(tag, vals, use_count);
        print_meta(tag, pt.depth(), pt.scale());
        if (ref_loaded_) {
            compare_and_report(tag, vals, pt.depth(), pt.scale(), use_count);
        }
    }

  private:
    void print_values(const std::string& tag,
                      const std::vector<double>& vals, int count) const
    {
        const int limit =
            std::min(count, static_cast<int>(vals.size()));
    }

    void print_meta(const std::string& tag, int level, double scale) const
    {
    }

    static std::vector<double> parse_list(const std::string& list)
    {
        std::vector<double> values;
        std::string token;
        std::stringstream ss(list);
        while (std::getline(ss, token, ',')) {
            std::stringstream ts(token);
            double val = 0.0;
            if (ts >> val) {
                values.push_back(val);
            }
        }
        return values;
    }

    static std::string escape_json(const std::string& input)
    {
        std::string out;
        out.reserve(input.size() + 8);
        for (char c : input) {
            switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
            }
        }
        return out;
    }

    void compare_and_report(const std::string& tag,
                            const std::vector<double>& gpu_vals, int level,
                            double scale, int count)
    {
        auto it = ref_.find(tag);
        if (it == ref_.end()) {
            return;
        }
        const std::vector<double>& cpu_vals = it->second;
        const int limit = std::min(
            count, static_cast<int>(std::min(cpu_vals.size(), gpu_vals.size())));
        double max_abs = 0.0;
        double max_rel = 0.0;
        int first_mismatch = -1;
        std::vector<std::pair<int, double>> abs_errors;
        abs_errors.reserve(static_cast<size_t>(limit));
        for (int i = 0; i < limit; ++i) {
            const double cpu = cpu_vals[static_cast<size_t>(i)];
            const double gpu = gpu_vals[static_cast<size_t>(i)];
            const double abs_err = std::fabs(cpu - gpu);
            const double denom = std::max(std::fabs(cpu), 1e-12);
            const double rel_err = abs_err / denom;
            if (abs_err > max_abs) {
                max_abs = abs_err;
            }
            if (rel_err > max_rel) {
                max_rel = rel_err;
            }
            if (first_mismatch < 0 &&
                (abs_err > tol_abs_ || rel_err > tol_rel_)) {
                first_mismatch = i;
            }
            abs_errors.emplace_back(i, abs_err);
        }
        std::sort(abs_errors.begin(), abs_errors.end(),
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;
                  });
        const int topn = std::min(5, static_cast<int>(abs_errors.size()));
        for (int i = 0; i < topn; ++i) {
            const int idx = abs_errors[static_cast<size_t>(i)].first;
            const double cpu = cpu_vals[static_cast<size_t>(idx)];
            const double gpu = gpu_vals[static_cast<size_t>(idx)];
            const double diff = cpu - gpu;
        }

        DiffEntry entry;
        entry.tag = tag;
        entry.max_abs = max_abs;
        entry.max_rel = max_rel;
        entry.idx = first_mismatch;
        entry.level = level;
        entry.scale = scale;
        entry.cpu.assign(cpu_vals.begin(),
                         cpu_vals.begin() + static_cast<size_t>(limit));
        entry.gpu.assign(gpu_vals.begin(),
                         gpu_vals.begin() + static_cast<size_t>(limit));
        diff_entries_.push_back(entry);

        if (first_bad_tag_.empty() &&
            (max_abs > tol_abs_ || max_rel > tol_rel_)) {
            first_bad_tag_ = tag;
        }
    }

    void write_json(const std::string& path) const
    {
        if (path.empty()) {
            return;
        }
        std::ofstream out(path);
        if (!out) {
            return;
        }
        out << "[\n";
        for (size_t i = 0; i < diff_entries_.size(); ++i) {
            const DiffEntry& e = diff_entries_[i];
            out << "  {\n";
            out << "    \"tag\": \"" << escape_json(e.tag) << "\",\n";
            out << "    \"max_abs\": " << e.max_abs << ",\n";
            out << "    \"max_rel\": " << e.max_rel << ",\n";
            out << "    \"idx\": " << e.idx << ",\n";
            out << "    \"level\": " << e.level << ",\n";
            out << "    \"scale\": " << e.scale << ",\n";
            out << "    \"cpu\": [";
            for (size_t j = 0; j < e.cpu.size(); ++j) {
                if (j > 0) {
                    out << ", ";
                }
                out << e.cpu[j];
            }
            out << "],\n";
            out << "    \"gpu\": [";
            for (size_t j = 0; j < e.gpu.size(); ++j) {
                if (j > 0) {
                    out << ", ";
                }
                out << e.gpu[j];
            }
            out << "]\n";
            out << "  }";
            if (i + 1 < diff_entries_.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
    }

    struct DiffEntry {
        std::string tag;
        double max_abs = 0.0;
        double max_rel = 0.0;
        int idx = -1;
        int level = -1;
        double scale = 0.0;
        std::vector<double> cpu;
        std::vector<double> gpu;
    };

    lowmem::FHEController* controller_ = nullptr;
    bool enabled_ = false;
    int count_ = 20;
    std::string layer_regex_;
    bool regex_valid_ = true;
    std::regex compiled_regex_;

    double tol_abs_ = 1e-3;
    double tol_rel_ = 1e-3;
    bool ref_loaded_ = false;
    std::unordered_map<std::string, std::vector<double>> ref_;
    std::vector<DiffEntry> diff_entries_;
    std::string first_bad_tag_;
};

#endif
