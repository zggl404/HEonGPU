#ifndef LOWMEM_RESNET20_DEBUG_DUMPER_H
#define LOWMEM_RESNET20_DEBUG_DUMPER_H

#include <algorithm>
#include <cstdio>
#include <string>
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

    bool should_dump(const std::string& tag) const
    {
        (void)tag;
        if (!enabled_) {
            return false;
        }
        return true;
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
    }

  private:
    void print_values(const std::string& tag,
                      const std::vector<double>& vals, int count) const
    {
        const int limit =
            std::min(count, static_cast<int>(vals.size()));
        std::printf("[DBG] %s: [ ", tag.c_str());
        for (int i = 0; i < limit; ++i) {
            std::printf("%7.3f", vals[static_cast<size_t>(i)]);
            if (i + 1 < limit) {
                std::printf(", ");
            }
        }
        std::printf(" ]\n");
    }

    void print_meta(const std::string& tag, int level, double scale) const
    {
        std::printf("[DBG_META] %s level=%d scale=%.6e\n", tag.c_str(), level,
                    scale);
    }

    lowmem::FHEController* controller_ = nullptr;
    bool enabled_ = false;
    int count_ = 20;
};

#endif
