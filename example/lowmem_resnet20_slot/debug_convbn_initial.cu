#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "DebugDumper.h"
#include "LowMemAdapter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using lowmem::Ctxt;
using lowmem::FHEController;
using lowmem::HEConfig;

static FHEController controller;
static DebugDumper dbg_dumper(&controller);

static std::string data_dir;
static std::string weights_dir;
static std::string input_filename = "luis.png";
static std::string reference_log_path;
static double tol_abs = 1e-2;
static double tol_rel = 1e-2;
static int dbg_count = 20;
static std::string diff_report_path =
    "output/diff_report_convbn_initial.json";

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
        } else if (arg == "--reference_log" && i + 1 < argc) {
            reference_log_path = argv[++i];
        } else if (arg == "--tol_abs" && i + 1 < argc) {
            tol_abs = std::atof(argv[++i]);
        } else if (arg == "--tol_rel" && i + 1 < argc) {
            tol_rel = std::atof(argv[++i]);
        } else if (arg == "--dbg_count" && i + 1 < argc) {
            dbg_count = std::atoi(argv[++i]);
        } else if (arg == "--diff_report" && i + 1 < argc) {
            diff_report_path = argv[++i];
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

static void print_vector_stats(const std::string& label,
                               const std::vector<double>& values)
{
    if (values.empty()) {
        std::cout << "[STATS] " << label << " size=0" << std::endl;
        return;
    }
    auto minmax = std::minmax_element(values.begin(), values.end());
    std::cout << "[STATS] " << label << " size=" << values.size()
              << " min=" << *minmax.first << " max=" << *minmax.second
              << " first5=[";
    const size_t limit = std::min<size_t>(5, values.size());
    for (size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << "]" << std::endl;
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
    std::cout << "[STATS] weights_dir=" << weights_dir << std::endl;

    HEConfig cfg;
    cfg.relu_degree = controller.relu_degree;
    controller.weights_dir = weights_dir;
    controller.initialize(cfg);

    dbg_dumper.set_enabled(true);
    dbg_dumper.set_count(dbg_count);
    dbg_dumper.set_tolerances(tol_abs, tol_rel);
    dbg_dumper.load_reference_log(reference_log_path);

    std::string input_path = data_dir + "/" + input_filename;
    std::vector<double> input_image = read_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Input image load failed. Check --data_dir and --input."
                  << std::endl;
        return 1;
    }

    Ctxt in = controller.encrypt(
        input_image,
        controller.circuit_depth - 4 -
            utils::get_relu_depth(controller.relu_degree));

    const std::string bias_path = weights_dir + "/conv1bn1-bias.bin";
    const std::string w_path = weights_dir + "/conv1bn1-ch0-k1.bin";
    std::cout << "[STATS] bias_path=" << bias_path
              << " text=" << (is_text_file(bias_path) ? "yes" : "no")
              << std::endl;
    std::cout << "[STATS] w_path=" << w_path
              << " text=" << (is_text_file(w_path) ? "yes" : "no") << std::endl;
    print_vector_stats("bias",
                       read_values_from_file(bias_path, 0.90));
    print_vector_stats("w_ch0_k1",
                       read_values_from_file(w_path, 0.90));

    dbg_dumper.dump_ct("Layer0/input encrypted (pre-initial_layer)", in);
    dbg_dumper.dump_ct("Layer0/initial_layer input (pre)", in);
    dbg_dumper.dump_ct("Initial layer input (pre)", in);

    double scale = 0.90;
    Ctxt res = controller.convbn_initial(in, scale, false);
    dbg_dumper.dump_ct("Initial layer convbn_initial (post)", res);

    std::filesystem::path report_path(diff_report_path);
    if (!report_path.empty() && report_path.has_parent_path()) {
        std::filesystem::create_directories(report_path.parent_path());
    }
    dbg_dumper.finalize(diff_report_path);

    return 0;
}
