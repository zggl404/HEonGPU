#ifndef LOWMEM_RESNET20_PLAIN_SIM_ADAPTER_H
#define LOWMEM_RESNET20_PLAIN_SIM_ADAPTER_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "Utils.h"

namespace plainsim {

using PlainVec = std::vector<double>;

class PlainSimController {
  public:
    int num_slots = 0;
    int base_slots = 0;
    int relu_degree = 119;
    bool relu_div_scale = true;
    std::string weights_dir;

    PlainVec pad_input(const PlainVec& values)
    {
        if (base_slots == 0) {
            base_slots = num_slots;
        }
        PlainVec out(static_cast<size_t>(base_slots), 0.0);
        int n = std::min(static_cast<int>(values.size()), base_slots);
        for (int i = 0; i < n; i++) {
            out[static_cast<size_t>(i)] = values[static_cast<size_t>(i)];
        }
        return out;
    }

    PlainVec encode_like(const PlainVec& values, int plaintext_num_slots)
    {
        if (base_slots == 0) {
            base_slots = num_slots;
        }
        PlainVec out(static_cast<size_t>(base_slots), 0.0);
        int n = std::min(plaintext_num_slots, base_slots);
        for (int i = 0; i < n; i++) {
            if (i < static_cast<int>(values.size())) {
                out[static_cast<size_t>(i)] = values[static_cast<size_t>(i)];
            }
        }
        return out;
    }

    PlainVec zero_vec() const
    {
        return PlainVec(static_cast<size_t>(base_slots), 0.0);
    }

    PlainVec read_values_scaled(const std::string& path, double scale)
    {
        return utils::read_values_from_file(path, scale);
    }

    PlainVec mult_scaled(const PlainVec& a, const PlainVec& b, double scale)
    {
        PlainVec out = a;
        const size_t n = std::min(out.size(), b.size());
        double inv = (scale == 0.0) ? 1.0 : (1.0 / scale);
        for (size_t i = 0; i < n; i++) {
            out[i] = out[i] * b[i] * inv;
        }
        return out;
    }

    PlainVec rotate_vector(const PlainVec& in, int steps)
    {
        const int n = static_cast<int>(in.size());
        if (n == 0) {
            return in;
        }
        int s = steps % n;
        if (s < 0) s += n;
        PlainVec out(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            out[static_cast<size_t>((i + s) % n)] = in[static_cast<size_t>(i)];
        }
        return out;
    }

    PlainVec add(const PlainVec& a, const PlainVec& b)
    {
        PlainVec out = a;
        const size_t n = std::min(out.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            out[i] += b[i];
        }
        return out;
    }

    PlainVec add_plain(const PlainVec& a, const PlainVec& b) { return add(a, b); }

    PlainVec mult(const PlainVec& a, double d)
    {
        PlainVec out = a;
        for (double& v : out) {
            v *= d;
        }
        return out;
    }

    PlainVec mult(const PlainVec& a, const PlainVec& b)
    {
        PlainVec out = a;
        const size_t n = std::min(out.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            out[i] *= b[i];
        }
        return out;
    }

    PlainVec mult_mask(const PlainVec& a, const PlainVec& mask)
    {
        return mult(a, mask);
    }

    PlainVec bootstrap(const PlainVec& in, bool /*timing*/ = false)
    {
        return in;
    }

    PlainVec relu(const PlainVec& in, double scale, bool /*timing*/ = false)
    {
        PlainVec out = in;
        for (double& v : out) {
            if (v <= 0.0) {
                v = 0.0;
            } else if (relu_div_scale) {
                v /= scale;
            }
        }
        return out;
    }

    PlainVec rotsum(const PlainVec& in, int slots)
    {
        PlainVec result = in;
        for (int i = 0; i < static_cast<int>(std::log2(slots)); i++) {
            result = add(result, rotate_vector(result, 1 << i));
        }
        return result;
    }

    PlainVec rotsum_padded(const PlainVec& in, int slots)
    {
        PlainVec result = in;
        for (int i = 0; i < static_cast<int>(std::log2(slots)); i++) {
            result = add(result, rotate_vector(result, slots * (1 << i)));
        }
        return result;
    }

    PlainVec repeat(const PlainVec& in, int slots)
    {
        return rotate_vector(rotsum(in, slots), -slots + 1);
    }

    PlainVec gen_mask(int n, int /*target_depth*/)
    {
        PlainVec mask(static_cast<size_t>(base_slots), 0.0);
        int copy_interval = n;
        for (int i = 0; i < num_slots; i++) {
            if (copy_interval > 0) {
                mask[static_cast<size_t>(i)] = 1.0;
            }
            copy_interval--;
            if (copy_interval <= -n) {
                copy_interval = n;
            }
        }
        return mask;
    }

    PlainVec mask_first_n(int n, int /*target_depth*/)
    {
        PlainVec mask(static_cast<size_t>(base_slots), 0.0);
        for (int i = 0; i < std::min(n, num_slots); i++) {
            mask[static_cast<size_t>(i)] = 1.0;
        }
        return mask;
    }

    PlainVec mask_second_n(int n, int /*target_depth*/)
    {
        PlainVec mask(static_cast<size_t>(base_slots), 0.0);
        int start = std::min(n, num_slots);
        for (int i = start; i < std::min(2 * n, num_slots); i++) {
            mask[static_cast<size_t>(i)] = 1.0;
        }
        return mask;
    }

    PlainVec mask_first_n_mod(int n, int mod, int shift, int /*target_depth*/)
    {
        PlainVec mask(static_cast<size_t>(base_slots), 0.0);
        for (int i = 0; i < num_slots; i++) {
            if ((i % mod) < n) {
                int idx = i + shift;
                if (idx < num_slots) {
                    mask[static_cast<size_t>(idx)] = 1.0;
                }
            }
        }
        return mask;
    }

    PlainVec mask_first_n_mod2(int n, int mod, int shift, int /*target_depth*/)
    {
        PlainVec mask(static_cast<size_t>(base_slots), 0.0);
        for (int i = 0; i < num_slots; i++) {
            if ((i % mod) < n) {
                int idx = i + shift;
                if (idx < num_slots) {
                    mask[static_cast<size_t>(idx)] = 1.0;
                }
            }
        }
        return mask;
    }

    PlainVec mask_channel(int n, int /*target_depth*/)
    {
        PlainVec mask;
        mask.reserve(static_cast<size_t>(base_slots));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 256; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 63 - n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        if (static_cast<int>(mask.size()) < base_slots) {
            mask.resize(static_cast<size_t>(base_slots), 0.0);
        }
        return mask;
    }

    PlainVec mask_channel_2(int n, int /*target_depth*/)
    {
        PlainVec mask;
        mask.reserve(static_cast<size_t>(base_slots));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 64; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 64; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 63 - n; i++) {
            for (int j = 0; j < 64; j++) {
                mask.push_back(0);
            }
        }

        if (static_cast<int>(mask.size()) < base_slots) {
            mask.resize(static_cast<size_t>(base_slots), 0.0);
        }
        return mask;
    }

    PlainVec mask_mod(int n, int /*target_depth*/, double custom_val)
    {
        PlainVec vec(static_cast<size_t>(base_slots), 0.0);
        for (int i = 0; i < num_slots; i++) {
            if (i % n == 0) {
                vec[static_cast<size_t>(i)] = custom_val;
            }
        }
        return vec;
    }

    PlainVec mask_from_to(int from, int to, int /*target_depth*/)
    {
        PlainVec vec(static_cast<size_t>(base_slots), 0.0);
        for (int i = from; i < std::min(to, num_slots); i++) {
            vec[static_cast<size_t>(i)] = 1.0;
        }
        return vec;
    }

    PlainVec convbn_initial(const PlainVec& in, double scale = 0.5,
                            bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 32;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias_values = read_values_scaled(
            weights_dir + "/conv1bn1-bias.bin", scale);
        PlainVec bias = encode_like(bias_values, 16384);

        PlainVec finalsum = zero_vec();
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/conv1bn1-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 16384);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
            }

            PlainVec sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            PlainVec res = add(sum, rotate_vector(sum, 1024));
            res = add(res, rotate_vector(rotate_vector(sum, 1024), 1024));
            res = mult_mask(res, mask_from_to(0, 1024, 0));

            if (!init) {
                finalsum = rotate_vector(res, 1024);
                init = true;
            } else {
                finalsum = add(finalsum, res);
                finalsum = rotate_vector(finalsum, 1024);
            }
        }

        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Initial layer");
        }

        return finalsum;
    }

    PlainVec convbn(const PlainVec& in, int layer, int n, double scale = 0.5,
                    bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 32;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        PlainVec finalsum = zero_vec();
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 16384);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
            }

            PlainVec sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            PlainVec res = add(sum, rotate_vector(sum, 1024));
            res = add(res, rotate_vector(rotate_vector(sum, 1024), 1024));
            res = mult_mask(res, mask_from_to(0, 1024, 0));

            if (!init) {
                finalsum = rotate_vector(res, 1024);
                init = true;
            } else {
                finalsum = add(finalsum, res);
                finalsum = rotate_vector(finalsum, 1024);
            }
        }

        PlainVec bias = encode_like(bias_values, 16384);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    PlainVec convbn2(const PlainVec& in, int layer, int n, double scale = 0.5,
                     bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 16;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        PlainVec finalsum = zero_vec();
        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 8192);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
            }

            PlainVec sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            if (!init) {
                finalsum = rotate_vector(sum, -256);
                init = true;
            } else {
                finalsum = add(finalsum, sum);
                finalsum = rotate_vector(finalsum, -256);
            }
        }

        PlainVec bias = encode_like(bias_values, 8192);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    PlainVec convbn3(const PlainVec& in, int layer, int n, double scale = 0.5,
                     bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 8;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        PlainVec finalsum = zero_vec();
        bool init = false;

        for (int j = 0; j < 64; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 4096);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
            }

            PlainVec sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            if (!init) {
                finalsum = rotate_vector(sum, -64);
                init = true;
            } else {
                finalsum = add(finalsum, sum);
                finalsum = rotate_vector(finalsum, -64);
            }
        }

        PlainVec bias = encode_like(bias_values, 4096);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    std::vector<PlainVec> convbn1632sx(const PlainVec& in, int layer, int n,
                                       double scale = 0.5,
                                       bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 16;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias1_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        PlainVec bias2_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        PlainVec finalSum016 = zero_vec();
        PlainVec finalSum1632 = zero_vec();

        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            std::vector<PlainVec> k_rows2;
            k_rows2.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec values2 = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 16) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 16384);
                PlainVec encoded2 = encode_like(values2, 16384);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
                k_rows2.push_back(mult_scaled(c_rotations[k], encoded2, scale));
            }

            PlainVec sum016 = k_rows[0];
            PlainVec sum1632 = k_rows2[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum016 = add(sum016, k_rows[i]);
                sum1632 = add(sum1632, k_rows2[i]);
            }

            PlainVec res016 = add(sum016, rotate_vector(sum016, 1024));
            res016 = add(res016, rotate_vector(rotate_vector(sum016, 1024),
                                               1024));
            res016 = mult_mask(res016, mask_from_to(0, 1024, 0));

            PlainVec res1632 = add(sum1632, rotate_vector(sum1632, 1024));
            res1632 = add(res1632, rotate_vector(rotate_vector(sum1632, 1024),
                                                 1024));
            res1632 = mult_mask(res1632, mask_from_to(0, 1024, 0));

            if (!init) {
                finalSum016 = rotate_vector(res016, 1024);
                finalSum1632 = rotate_vector(res1632, 1024);
                init = true;
            } else {
                finalSum016 = add(finalSum016, res016);
                finalSum016 = rotate_vector(finalSum016, 1024);
                finalSum1632 = add(finalSum1632, res1632);
                finalSum1632 = rotate_vector(finalSum1632, 1024);
            }
        }

        PlainVec bias1 = encode_like(bias1_values, 16384);
        PlainVec bias2 = encode_like(bias2_values, 16384);
        finalSum016 = add_plain(finalSum016, bias1);
        finalSum1632 = add_plain(finalSum1632, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnSx" +
                                             std::to_string(n));
        }

        return {finalSum016, finalSum1632};
    }

    std::vector<PlainVec> convbn1632dx(const PlainVec& in, int layer, int n,
                                       double scale = 0.5,
                                       bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 16;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, img_width));

        PlainVec bias1_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        PlainVec bias2_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        PlainVec finalSum016 = zero_vec();
        PlainVec finalSum1632 = zero_vec();

        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(3);
            std::vector<PlainVec> k_rows2;
            k_rows2.reserve(3);
            for (int k = 0; k < 3; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(1) + ".bin",
                    scale);
                PlainVec values2 = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 16) + "-k" +
                        std::to_string(1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 16384);
                PlainVec encoded2 = encode_like(values2, 16384);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
                k_rows2.push_back(mult_scaled(c_rotations[k], encoded2, scale));
            }

            PlainVec sum016 = k_rows[0];
            PlainVec sum1632 = k_rows2[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum016 = add(sum016, k_rows[i]);
                sum1632 = add(sum1632, k_rows2[i]);
            }

            if (!init) {
                finalSum016 = rotate_vector(sum016, -1024);
                finalSum1632 = rotate_vector(sum1632, -1024);
                init = true;
            } else {
                finalSum016 = add(finalSum016, sum016);
                finalSum016 = rotate_vector(finalSum016, -1024);
                finalSum1632 = add(finalSum1632, sum1632);
                finalSum1632 = rotate_vector(finalSum1632, -1024);
            }
        }

        PlainVec bias1 = encode_like(bias1_values, 16384);
        PlainVec bias2 = encode_like(bias2_values, 16384);
        finalSum016 = add_plain(finalSum016, bias1);
        finalSum1632 = add_plain(finalSum1632, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnDx" +
                                             std::to_string(n));
        }

        return {finalSum016, finalSum1632};
    }

    std::vector<PlainVec> convbn3264sx(const PlainVec& in, int layer, int n,
                                       double scale = 0.5,
                                       bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 8;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        PlainVec bias1_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        PlainVec bias2_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        PlainVec finalSum032 = zero_vec();
        PlainVec finalSum3264 = zero_vec();

        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(9);
            std::vector<PlainVec> k_rows2;
            k_rows2.reserve(9);
            for (int k = 0; k < 9; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec values2 = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 32) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 8192);
                PlainVec encoded2 = encode_like(values2, 8192);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
                k_rows2.push_back(mult_scaled(c_rotations[k], encoded2, scale));
            }

            PlainVec sum032 = k_rows[0];
            PlainVec sum3264 = k_rows2[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum032 = add(sum032, k_rows[i]);
                sum3264 = add(sum3264, k_rows2[i]);
            }

            if (!init) {
                finalSum032 = rotate_vector(sum032, -256);
                finalSum3264 = rotate_vector(sum3264, -256);
                init = true;
            } else {
                finalSum032 = add(finalSum032, sum032);
                finalSum032 = rotate_vector(finalSum032, -256);
                finalSum3264 = add(finalSum3264, sum3264);
                finalSum3264 = rotate_vector(finalSum3264, -256);
            }
        }

        PlainVec bias1 = encode_like(bias1_values, 8192);
        PlainVec bias2 = encode_like(bias2_values, 8192);
        finalSum032 = add_plain(finalSum032, bias1);
        finalSum3264 = add_plain(finalSum3264, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnSx" +
                                             std::to_string(n));
        }

        return {finalSum032, finalSum3264};
    }

    std::vector<PlainVec> convbn3264dx(const PlainVec& in, int layer, int n,
                                       double scale = 0.5,
                                       bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 8;
        int padding = 1;

        std::vector<PlainVec> c_rotations;
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, img_width));

        PlainVec bias1_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        PlainVec bias2_values = read_values_scaled(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        PlainVec finalSum032 = zero_vec();
        PlainVec finalSum3264 = zero_vec();

        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<PlainVec> k_rows;
            k_rows.reserve(3);
            std::vector<PlainVec> k_rows2;
            k_rows2.reserve(3);
            for (int k = 0; k < 3; k++) {
                PlainVec values = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(1) + ".bin",
                    scale);
                PlainVec values2 = read_values_scaled(
                    weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 32) + "-k" +
                        std::to_string(1) + ".bin",
                    scale);
                PlainVec encoded = encode_like(values, 8192);
                PlainVec encoded2 = encode_like(values2, 8192);
                k_rows.push_back(mult_scaled(c_rotations[k], encoded, scale));
                k_rows2.push_back(mult_scaled(c_rotations[k], encoded2, scale));
            }

            PlainVec sum032 = k_rows[0];
            PlainVec sum3264 = k_rows2[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum032 = add(sum032, k_rows[i]);
                sum3264 = add(sum3264, k_rows2[i]);
            }

            if (!init) {
                finalSum032 = rotate_vector(sum032, -256);
                finalSum3264 = rotate_vector(sum3264, -256);
                init = true;
            } else {
                finalSum032 = add(finalSum032, sum032);
                finalSum032 = rotate_vector(finalSum032, -256);
                finalSum3264 = add(finalSum3264, sum3264);
                finalSum3264 = rotate_vector(finalSum3264, -256);
            }
        }

        PlainVec bias1 = encode_like(bias1_values, 8192);
        PlainVec bias2 = encode_like(bias2_values, 8192);
        finalSum032 = add_plain(finalSum032, bias1);
        finalSum3264 = add_plain(finalSum3264, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnDx" +
                                             std::to_string(n));
        }

        return {finalSum032, finalSum3264};
    }

    PlainVec downsample1024to256(const PlainVec& c1, const PlainVec& c2)
    {
        num_slots = 16384 * 2;

        PlainVec fullpack = add(mult_mask(c1, mask_first_n(16384, 0)),
                                mult_mask(c2, mask_second_n(16384, 0)));

        fullpack = mult_mask(add(fullpack, rotate_vector(fullpack, 1)),
                             gen_mask(2, 0));
        fullpack =
            mult_mask(add(fullpack, rotate_vector(rotate_vector(fullpack, 1), 1)),
                      gen_mask(4, 0));
        fullpack =
            mult_mask(add(fullpack, rotate_vector(fullpack, 4)), gen_mask(8, 0));
        fullpack = add(fullpack, rotate_vector(fullpack, 8));

        PlainVec downsampledrows = zero_vec();

        for (int i = 0; i < 16; i++) {
            PlainVec masked = mult_mask(
                fullpack, mask_first_n_mod(16, 1024, i, 0));
            downsampledrows = add(downsampledrows, masked);
            if (i < 15) {
                fullpack = rotate_vector(fullpack, 64 - 16);
            }
        }

        PlainVec downsampledchannels = zero_vec();
        for (int i = 0; i < 32; i++) {
            PlainVec masked =
                mult_mask(downsampledrows, mask_channel(i, 0));
            downsampledchannels = add(downsampledchannels, masked);
            downsampledchannels =
                rotate_vector(downsampledchannels, -(1024 - 256));
        }

        downsampledchannels =
            rotate_vector(downsampledchannels, (1024 - 256) * 32);
        downsampledchannels =
            add(downsampledchannels, rotate_vector(downsampledchannels, -8192));
        downsampledchannels = add(
            downsampledchannels,
            rotate_vector(rotate_vector(downsampledchannels, -8192), -8192));

        num_slots = 8192;
        return downsampledchannels;
    }

    PlainVec downsample256to64(const PlainVec& c1, const PlainVec& c2)
    {
        num_slots = 8192 * 2;
        PlainVec fullpack = add(mult_mask(c1, mask_first_n(8192, 0)),
                                mult_mask(c2, mask_second_n(8192, 0)));

        fullpack = mult_mask(add(fullpack, rotate_vector(fullpack, 1)),
                             gen_mask(2, 0));
        fullpack =
            mult_mask(add(fullpack, rotate_vector(rotate_vector(fullpack, 1), 1)),
                      gen_mask(4, 0));
        fullpack = add(fullpack, rotate_vector(fullpack, 4));

        PlainVec downsampledrows = zero_vec();

        for (int i = 0; i < 32; i++) {
            PlainVec masked =
                mult_mask(fullpack, mask_first_n_mod2(8, 256, i, 0));
            downsampledrows = add(downsampledrows, masked);
            if (i < 31) {
                fullpack = rotate_vector(fullpack, 32 - 8);
            }
        }

        PlainVec downsampledchannels = zero_vec();
        for (int i = 0; i < 64; i++) {
            PlainVec masked =
                mult_mask(downsampledrows, mask_channel_2(i, 0));
            downsampledchannels = add(downsampledchannels, masked);
            downsampledchannels =
                rotate_vector(downsampledchannels, -(256 - 64));
        }

        downsampledchannels =
            rotate_vector(downsampledchannels, (256 - 64) * 64);
        downsampledchannels =
            add(downsampledchannels, rotate_vector(downsampledchannels, -4096));
        downsampledchannels = add(
            downsampledchannels,
            rotate_vector(rotate_vector(downsampledchannels, -4096), -4096));

        num_slots = 4096;
        return downsampledchannels;
    }
};

} // namespace plainsim

#endif // LOWMEM_RESNET20_PLAIN_SIM_ADAPTER_H
