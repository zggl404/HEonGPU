#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "PlainSimAdapter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using plainsim::PlainSimController;
using plainsim::PlainVec;

static PlainSimController plain_controller;

static std::string data_dir;
static std::string weights_dir;
static std::string input_filename = "luis.png";
static bool relu_div_scale = true;
static int verbose = 0;

static void parse_args(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--weights_dir" && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_filename = argv[++i];
        } else if (arg == "--relu_div_scale") {
            relu_div_scale = true;
        } else if (arg == "--no_relu_div_scale") {
            relu_div_scale = false;
        } else if (arg == "--verbose" && i + 1 < argc) {
            verbose = std::atoi(argv[++i]);
        }
    }
}

static std::filesystem::path resolve_path(const std::filesystem::path& base,
                                          const std::string& path)
{
    if (path.empty()) {
        return {};
    }
    std::filesystem::path p(path);
    if (p.is_absolute()) {
        return p;
    }
    return base / p;
}

static std::vector<double> read_image(const std::string& filename)
{
    int width = 32;
    int height = 32;
    int channels = 3;
    unsigned char* image_data =
        stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Could not load the image in " << filename << std::endl;
        return {};
    }

    std::vector<double> imageVector;
    imageVector.reserve(width * height * channels);

    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0);
    }
    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(
            static_cast<double>(image_data[1 + 3 * i]) / 255.0);
    }
    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(
            static_cast<double>(image_data[2 + 3 * i]) / 255.0);
    }

    stbi_image_free(image_data);
    return imageVector;
}

static PlainVec initial_layer_plain(const PlainVec& in)
{
    double scale = 0.90;
    PlainVec res = plain_controller.convbn_initial(in, scale, verbose > 1);
    res = plain_controller.bootstrap(res, verbose > 1);
    res = plain_controller.relu(res, scale, verbose > 1);
    return res;
}

static PlainVec layer1_plain(const PlainVec& in)
{
    bool timing = verbose > 1;
    double scale = 1.00;

    PlainVec res1 = plain_controller.convbn(in, 1, 1, scale, timing);
    res1 = plain_controller.bootstrap(res1, timing);
    res1 = plain_controller.relu(res1, scale, timing);

    scale = 0.52;
    res1 = plain_controller.convbn(res1, 1, 2, scale, timing);
    res1 = plain_controller.add(res1, plain_controller.mult(in, scale));
    res1 = plain_controller.bootstrap(res1, timing);
    res1 = plain_controller.relu(res1, scale, timing);

    scale = 0.55;
    PlainVec res2 = plain_controller.convbn(res1, 2, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.36;
    res2 = plain_controller.convbn(res2, 2, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.63;
    PlainVec res3 = plain_controller.convbn(res2, 3, 1, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    scale = 0.42;
    res3 = plain_controller.convbn(res3, 3, 2, scale, timing);
    res3 = plain_controller.add(res3, plain_controller.mult(res2, scale));
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    return res3;
}

static PlainVec layer2_plain(const PlainVec& in)
{
    double scaleSx = 0.57;
    double scaleDx = 0.40;
    bool timing = verbose > 1;

    PlainVec boot_in = plain_controller.bootstrap(in, timing);

    std::vector<PlainVec> res1sx =
        plain_controller.convbn1632sx(boot_in, 4, 1, scaleSx, timing);
    std::vector<PlainVec> res1dx =
        plain_controller.convbn1632dx(boot_in, 4, 1, scaleDx, timing);

    PlainVec fullpackSx =
        plain_controller.downsample1024to256(res1sx[0], res1sx[1]);
    PlainVec fullpackDx =
        plain_controller.downsample1024to256(res1dx[0], res1dx[1]);

    plain_controller.num_slots = 8192;
    fullpackSx = plain_controller.bootstrap(fullpackSx, timing);

    fullpackSx = plain_controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = plain_controller.convbn2(fullpackSx, 4, 2, scaleDx, timing);

    PlainVec res1 = plain_controller.bootstrap(fullpackDx, timing);
    res1 = plain_controller.relu(res1, scaleDx, timing);
    res1 = plain_controller.add(res1, fullpackSx);

    double scale = 0.76;
    PlainVec res2 = plain_controller.convbn2(res1, 5, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.37;
    res2 = plain_controller.convbn2(res2, 5, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.63;
    PlainVec res3 = plain_controller.convbn2(res2, 6, 1, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    scale = 0.25;
    res3 = plain_controller.convbn2(res3, 6, 2, scale, timing);
    res3 = plain_controller.add(res3, plain_controller.mult(res2, scale));
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    return res3;
}

static PlainVec layer3_plain(const PlainVec& in)
{
    double scaleSx = 0.63;
    double scaleDx = 0.40;
    bool timing = verbose > 1;

    PlainVec boot_in = plain_controller.bootstrap(in, timing);

    std::vector<PlainVec> res1sx =
        plain_controller.convbn3264sx(boot_in, 7, 1, scaleSx, timing);
    std::vector<PlainVec> res1dx =
        plain_controller.convbn3264dx(boot_in, 7, 1, scaleDx, timing);

    PlainVec fullpackSx =
        plain_controller.downsample256to64(res1sx[0], res1sx[1]);
    PlainVec fullpackDx =
        plain_controller.downsample256to64(res1dx[0], res1dx[1]);

    plain_controller.num_slots = 4096;
    fullpackSx = plain_controller.bootstrap(fullpackSx, timing);

    fullpackSx = plain_controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = plain_controller.convbn3(fullpackSx, 7, 2, scaleDx, timing);

    PlainVec res1 = plain_controller.bootstrap(fullpackDx, timing);
    res1 = plain_controller.relu(res1, scaleDx, timing);
    res1 = plain_controller.add(res1, fullpackSx);

    double scale = 0.57;
    PlainVec res2 = plain_controller.convbn3(res1, 8, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.33;
    res2 = plain_controller.convbn3(res2, 8, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.69;
    PlainVec res3 = plain_controller.convbn3(res2, 9, 1, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    scale = 0.10;
    res3 = plain_controller.convbn3(res3, 9, 2, scale, timing);
    res3 = plain_controller.add(res3, plain_controller.mult(res2, scale));
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);

    return res3;
}

static PlainVec final_layer_plain(const PlainVec& in)
{
    plain_controller.num_slots = 4096;

    std::vector<double> fc_weight =
        utils::read_fc_weight(weights_dir + "/fc.bin");
    PlainVec weight =
        plain_controller.encode_like(fc_weight, plain_controller.num_slots);

    PlainVec res = plain_controller.rotsum(in, 64);
    res = plain_controller.mult(
        res, plain_controller.mask_mod(64, 0, 1.0 / 64.0));

    res = plain_controller.repeat(res, 16);
    res = plain_controller.mult(res, weight);
    res = plain_controller.rotsum_padded(res, 64);

    return res;
}

int main(int argc, char* argv[])
{
    parse_args(argc, argv);

    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
    std::filesystem::path repo_root =
        exe_path.parent_path() / ".." / ".." / "..";
    repo_root = std::filesystem::weakly_canonical(repo_root);

    if (data_dir.empty()) {
        data_dir =
            (repo_root / "example/lowmem_resnet20_slot/assets/inputs").string();
    } else {
        data_dir = resolve_path(repo_root, data_dir).string();
    }

    if (weights_dir.empty()) {
        weights_dir =
            (repo_root / "example/lowmem_resnet20_slot/assets/weights").string();
    } else {
        weights_dir = resolve_path(repo_root, weights_dir).string();
    }

    plain_controller.weights_dir = weights_dir;
    plain_controller.num_slots = 32768;
    plain_controller.relu_div_scale = relu_div_scale;

    std::string input_path = data_dir + "/" + input_filename;
    std::vector<double> input_image = read_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Input image load failed. Check --data_dir and --input."
                  << std::endl;
        return 1;
    }

    PlainVec in = plain_controller.pad_input(input_image);

    PlainVec first = initial_layer_plain(in);
    PlainVec l1 = layer1_plain(first);
    PlainVec l2 = layer2_plain(l1);
    PlainVec l3 = layer3_plain(l2);
    PlainVec out = final_layer_plain(l3);

    std::cout << "PlainSim output:[";
    for (int i = 0; i < 10; i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(3)
                  << out[static_cast<size_t>(i)];
    }
    std::cout << "]" << std::endl;

    auto max_it = std::max_element(out.begin(), out.begin() + 10);
    int index_max = static_cast<int>(std::distance(out.begin(), max_it));
    std::cout << "PlainSim top-1: " << utils::get_class(index_max)
              << " (" << index_max << ")" << std::endl;

    return 0;
}
