#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "LowMemAdapter.h"
#include "PlainSimAdapter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN_TEXT ""

using lowmem::Ctxt;
using lowmem::FHEController;
using lowmem::HEConfig;

static FHEController controller;
static plainsim::PlainSimController plain_controller;

static std::string data_dir;
static std::string weights_dir;
static std::string input_filename = "luis.png";
static int verbose = 0;
static bool plain = false;
static bool debug_cuda = false;
static bool debug_encode = false;
static bool debug_plain = false;
static bool plain_sim = false;
static bool compare_plain_sim = false;

static void debug_decrypt(const std::string& label, const Ctxt& c, int slots);

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
        } else if (arg == "--verbose" && i + 1 < argc) {
            verbose = std::atoi(argv[++i]);
        } else if (arg == "--plain") {
            plain = true;
        } else if (arg == "--debug_cuda") {
            debug_cuda = true;
        } else if (arg == "--debug_encode") {
            debug_encode = true;
        } else if (arg == "--debug_plain") {
            debug_plain = true;
        } else if (arg == "--plain_sim") {
            plain_sim = true;
        } else if (arg == "--compare_plain_sim") {
            compare_plain_sim = true;
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

static Ctxt initial_layer(const Ctxt& in)
{
    double scale = 0.90;
    Ctxt res = controller.convbn_initial(in, scale, verbose > 1);
    if (debug_plain) {
        debug_decrypt("Initial convbn out", res, 8);
    }
    res = controller.bootstrap(res, verbose > 1);
    if (debug_plain) {
        debug_decrypt("Initial bootstrap out", res, 8);
    }
    if (debug_plain) {
        debug_decrypt("Initial relu in", res, 8);
    }
    res = controller.relu(res, scale, verbose > 1);
    if (debug_plain) {
        debug_decrypt("Initial relu out", res, 8);
    }
    return res;
}
static void debug_decrypt(const std::string& label, const Ctxt& c, int slots)
{
    std::vector<double> v = controller.decrypt_tovector(c, slots);
    auto minmax = std::minmax_element(v.begin(), v.end());
    std::cout << label << " min=" << std::fixed << std::setprecision(6)
              << *minmax.first << " max=" << *minmax.second << " head=[";
    size_t head = std::min<size_t>(v.size(), 8);
    for (size_t i = 0; i < head; i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << v[i];
    }
    std::cout << "]" << std::endl;
}

static double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b)
{
    size_t n = std::min(a.size(), b.size());
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static void compare_plain_sim_layer(const std::string& label,
                                    const plainsim::PlainVec& plain,
                                    const Ctxt& c, int slots)
{
    std::vector<double> dec = controller.decrypt_tovector(c, slots);
    double max_err = max_abs_diff(plain, dec);
    std::cout << label << " max_abs=" << std::setprecision(6) << max_err
              << std::endl;
}

static plainsim::PlainVec initial_layer_plain(const plainsim::PlainVec& in)
{
    double scale = 0.90;
    plainsim::PlainVec res =
        plain_controller.convbn_initial(in, scale, verbose > 1);
    res = plain_controller.bootstrap(res, verbose > 1);
    res = plain_controller.relu(res, scale, verbose > 1);
    return res;
}

static plainsim::PlainVec layer1_plain(const plainsim::PlainVec& in)
{
    bool timing = verbose > 1;
    double scale = 1.00;

    plainsim::PlainVec res1 = plain_controller.convbn(in, 1, 1, scale, timing);
    res1 = plain_controller.bootstrap(res1, timing);
    res1 = plain_controller.relu(res1, scale, timing);

    scale = 0.52;

    res1 = plain_controller.convbn(res1, 1, 2, scale, timing);
    res1 = plain_controller.add(res1, plain_controller.mult(in, scale));
    res1 = plain_controller.bootstrap(res1, timing);
    res1 = plain_controller.relu(res1, scale, timing);

    scale = 0.55;

    plainsim::PlainVec res2 = plain_controller.convbn(res1, 2, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.36;

    res2 = plain_controller.convbn(res2, 2, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.63;

    plainsim::PlainVec res3 = plain_controller.convbn(res2, 3, 1, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    scale = 0.42;

    res3 = plain_controller.convbn(res3, 3, 2, scale, timing);
    res3 = plain_controller.add(res3, plain_controller.mult(res2, scale));
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    return res3;
}

static plainsim::PlainVec layer2_plain(const plainsim::PlainVec& in)
{
    double scaleSx = 0.57;
    double scaleDx = 0.40;
    bool timing = verbose > 1;

    plainsim::PlainVec boot_in = plain_controller.bootstrap(in, timing);

    std::vector<plainsim::PlainVec> res1sx =
        plain_controller.convbn1632sx(boot_in, 4, 1, scaleSx, timing);
    std::vector<plainsim::PlainVec> res1dx =
        plain_controller.convbn1632dx(boot_in, 4, 1, scaleDx, timing);

    plainsim::PlainVec fullpackSx =
        plain_controller.downsample1024to256(res1sx[0], res1sx[1]);
    plainsim::PlainVec fullpackDx =
        plain_controller.downsample1024to256(res1dx[0], res1dx[1]);

    plain_controller.num_slots = 8192;
    fullpackSx = plain_controller.bootstrap(fullpackSx, timing);

    fullpackSx = plain_controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = plain_controller.convbn2(fullpackSx, 4, 2, scaleDx, timing);

    plainsim::PlainVec res1 = plain_controller.bootstrap(fullpackDx, timing);
    res1 = plain_controller.relu(res1, scaleDx, timing);
    res1 = plain_controller.add(res1, fullpackSx);

    double scale = 0.76;
    plainsim::PlainVec res2 = plain_controller.convbn2(res1, 5, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.37;
    res2 = plain_controller.convbn2(res2, 5, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.63;
    plainsim::PlainVec res3 = plain_controller.convbn2(res2, 6, 1, scale, timing);
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    scale = 0.25;
    res3 = plain_controller.convbn2(res3, 6, 2, scale, timing);
    res3 = plain_controller.add(res3, plain_controller.mult(res2, scale));
    res3 = plain_controller.bootstrap(res3, timing);
    res3 = plain_controller.relu(res3, scale, timing);

    return res3;
}

static Ctxt layer2_compare(const Ctxt& in_ct, const plainsim::PlainVec& in_plain)
{
    double scaleSx = 0.57;
    double scaleDx = 0.40;
    bool timing = verbose > 1;

    Ctxt boot_in_ct = controller.bootstrap(in_ct, timing);
    plainsim::PlainVec boot_in_plain = plain_controller.bootstrap(in_plain, timing);

    std::vector<Ctxt> res1sx_ct =
        controller.convbn1632sx(boot_in_ct, 4, 1, scaleSx, timing);
    std::vector<Ctxt> res1dx_ct =
        controller.convbn1632dx(boot_in_ct, 4, 1, scaleDx, timing);

    std::vector<plainsim::PlainVec> res1sx_plain =
        plain_controller.convbn1632sx(boot_in_plain, 4, 1, scaleSx, timing);
    std::vector<plainsim::PlainVec> res1dx_plain =
        plain_controller.convbn1632dx(boot_in_plain, 4, 1, scaleDx, timing);

    Ctxt fullpackSx_ct = controller.downsample1024to256(res1sx_ct[0], res1sx_ct[1]);
    Ctxt fullpackDx_ct = controller.downsample1024to256(res1dx_ct[0], res1dx_ct[1]);
    plainsim::PlainVec fullpackSx_plain =
        plain_controller.downsample1024to256(res1sx_plain[0], res1sx_plain[1]);
    plainsim::PlainVec fullpackDx_plain =
        plain_controller.downsample1024to256(res1dx_plain[0], res1dx_plain[1]);

    compare_plain_sim_layer("Layer2 after downsample Sx", fullpackSx_plain,
                            fullpackSx_ct, plain_controller.num_slots);
    compare_plain_sim_layer("Layer2 after downsample Dx", fullpackDx_plain,
                            fullpackDx_ct, plain_controller.num_slots);

    controller.num_slots = 8192;
    plain_controller.num_slots = 8192;

    fullpackSx_ct = controller.bootstrap(fullpackSx_ct, timing);
    fullpackSx_plain = plain_controller.bootstrap(fullpackSx_plain, timing);
    compare_plain_sim_layer("Layer2 after boot Sx", fullpackSx_plain,
                            fullpackSx_ct, plain_controller.num_slots);

    fullpackSx_ct = controller.relu(fullpackSx_ct, scaleSx, timing);
    fullpackSx_plain = plain_controller.relu(fullpackSx_plain, scaleSx, timing);
    compare_plain_sim_layer("Layer2 after relu Sx", fullpackSx_plain,
                            fullpackSx_ct, plain_controller.num_slots);

    fullpackSx_ct = controller.convbn2(fullpackSx_ct, 4, 2, scaleDx, timing);
    fullpackSx_plain =
        plain_controller.convbn2(fullpackSx_plain, 4, 2, scaleDx, timing);
    compare_plain_sim_layer("Layer2 after convbn2 Sx", fullpackSx_plain,
                            fullpackSx_ct, plain_controller.num_slots);

    Ctxt res1_ct = controller.bootstrap(fullpackDx_ct, timing);
    plainsim::PlainVec res1_plain = plain_controller.bootstrap(fullpackDx_plain, timing);
    compare_plain_sim_layer("Layer2 after boot Dx", res1_plain, res1_ct,
                            plain_controller.num_slots);

    res1_ct = controller.relu(res1_ct, scaleDx, timing);
    res1_plain = plain_controller.relu(res1_plain, scaleDx, timing);
    compare_plain_sim_layer("Layer2 after relu Dx", res1_plain, res1_ct,
                            plain_controller.num_slots);

    res1_ct = controller.add(res1_ct, fullpackSx_ct);
    res1_plain = plain_controller.add(res1_plain, fullpackSx_plain);
    compare_plain_sim_layer("Layer2 after add", res1_plain, res1_ct,
                            plain_controller.num_slots);

    double scale = 0.76;
    Ctxt res2_ct = controller.convbn2(res1_ct, 5, 1, scale, timing);
    plainsim::PlainVec res2_plain =
        plain_controller.convbn2(res1_plain, 5, 1, scale, timing);
    compare_plain_sim_layer("Layer2 after convbn2(5,1)", res2_plain, res2_ct,
                            plain_controller.num_slots);

    res2_ct = controller.bootstrap(res2_ct, timing);
    res2_plain = plain_controller.bootstrap(res2_plain, timing);
    compare_plain_sim_layer("Layer2 after boot res2", res2_plain, res2_ct,
                            plain_controller.num_slots);

    res2_ct = controller.relu(res2_ct, scale, timing);
    res2_plain = plain_controller.relu(res2_plain, scale, timing);
    compare_plain_sim_layer("Layer2 after relu res2", res2_plain, res2_ct,
                            plain_controller.num_slots);

    scale = 0.37;
    res2_ct = controller.convbn2(res2_ct, 5, 2, scale, timing);
    res2_plain = plain_controller.convbn2(res2_plain, 5, 2, scale, timing);
    compare_plain_sim_layer("Layer2 after convbn2(5,2)", res2_plain, res2_ct,
                            plain_controller.num_slots);

    res2_ct = controller.add(res2_ct, controller.mult(res1_ct, scale));
    res2_plain = plain_controller.add(res2_plain,
                                      plain_controller.mult(res1_plain, scale));
    compare_plain_sim_layer("Layer2 after add skip1", res2_plain, res2_ct,
                            plain_controller.num_slots);

    res2_ct = controller.bootstrap(res2_ct, timing);
    res2_plain = plain_controller.bootstrap(res2_plain, timing);
    compare_plain_sim_layer("Layer2 after boot res2b", res2_plain, res2_ct,
                            plain_controller.num_slots);

    res2_ct = controller.relu(res2_ct, scale, timing);
    res2_plain = plain_controller.relu(res2_plain, scale, timing);
    compare_plain_sim_layer("Layer2 after relu res2b", res2_plain, res2_ct,
                            plain_controller.num_slots);

    scale = 0.63;
    Ctxt res3_ct = controller.convbn2(res2_ct, 6, 1, scale, timing);
    plainsim::PlainVec res3_plain =
        plain_controller.convbn2(res2_plain, 6, 1, scale, timing);
    compare_plain_sim_layer("Layer2 after convbn2(6,1)", res3_plain, res3_ct,
                            plain_controller.num_slots);

    res3_ct = controller.bootstrap(res3_ct, timing);
    res3_plain = plain_controller.bootstrap(res3_plain, timing);
    compare_plain_sim_layer("Layer2 after boot res3", res3_plain, res3_ct,
                            plain_controller.num_slots);

    res3_ct = controller.relu(res3_ct, scale, timing);
    res3_plain = plain_controller.relu(res3_plain, scale, timing);
    compare_plain_sim_layer("Layer2 after relu res3", res3_plain, res3_ct,
                            plain_controller.num_slots);

    scale = 0.25;
    res3_ct = controller.convbn2(res3_ct, 6, 2, scale, timing);
    res3_plain = plain_controller.convbn2(res3_plain, 6, 2, scale, timing);
    compare_plain_sim_layer("Layer2 after convbn2(6,2)", res3_plain, res3_ct,
                            plain_controller.num_slots);

    res3_ct = controller.add(res3_ct, controller.mult(res2_ct, scale));
    res3_plain = plain_controller.add(res3_plain,
                                      plain_controller.mult(res2_plain, scale));
    compare_plain_sim_layer("Layer2 after add skip2", res3_plain, res3_ct,
                            plain_controller.num_slots);

    res3_ct = controller.bootstrap(res3_ct, timing);
    res3_plain = plain_controller.bootstrap(res3_plain, timing);
    compare_plain_sim_layer("Layer2 after boot out", res3_plain, res3_ct,
                            plain_controller.num_slots);

    res3_ct = controller.relu(res3_ct, scale, timing);
    res3_plain = plain_controller.relu(res3_plain, scale, timing);
    compare_plain_sim_layer("Layer2 after relu out", res3_plain, res3_ct,
                            plain_controller.num_slots);

    return res3_ct;
}

static plainsim::PlainVec layer3_plain(const plainsim::PlainVec& in)
{
    double scaleSx = 0.63;
    double scaleDx = 0.40;
    bool timing = verbose > 1;

    plainsim::PlainVec boot_in = plain_controller.bootstrap(in, timing);

    std::vector<plainsim::PlainVec> res1sx =
        plain_controller.convbn3264sx(boot_in, 7, 1, scaleSx, timing);
    std::vector<plainsim::PlainVec> res1dx =
        plain_controller.convbn3264dx(boot_in, 7, 1, scaleDx, timing);

    plainsim::PlainVec fullpackSx =
        plain_controller.downsample256to64(res1sx[0], res1sx[1]);
    plainsim::PlainVec fullpackDx =
        plain_controller.downsample256to64(res1dx[0], res1dx[1]);

    plain_controller.num_slots = 4096;
    fullpackSx = plain_controller.bootstrap(fullpackSx, timing);

    fullpackSx = plain_controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = plain_controller.convbn3(fullpackSx, 7, 2, scaleDx, timing);

    plainsim::PlainVec res1 = plain_controller.bootstrap(fullpackDx, timing);
    res1 = plain_controller.relu(res1, scaleDx, timing);
    res1 = plain_controller.add(res1, fullpackSx);

    double scale = 0.57;
    plainsim::PlainVec res2 = plain_controller.convbn3(res1, 8, 1, scale, timing);
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.33;
    res2 = plain_controller.convbn3(res2, 8, 2, scale, timing);
    res2 = plain_controller.add(res2, plain_controller.mult(res1, scale));
    res2 = plain_controller.bootstrap(res2, timing);
    res2 = plain_controller.relu(res2, scale, timing);

    scale = 0.69;
    plainsim::PlainVec res3 = plain_controller.convbn3(res2, 9, 1, scale, timing);
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

static plainsim::PlainVec final_layer_plain(const plainsim::PlainVec& in)
{
    plain_controller.num_slots = 4096;

    std::vector<double> fc_weight =
        utils::read_fc_weight(weights_dir + "/fc.bin");
    plainsim::PlainVec weight =
        plain_controller.encode_like(fc_weight, plain_controller.num_slots);

    plainsim::PlainVec res = plain_controller.rotsum(in, 64);
    res = plain_controller.mult(
        res, plain_controller.mask_mod(64, 0, 1.0 / 64.0));

    res = plain_controller.repeat(res, 16);
    res = plain_controller.mult(res, weight);
    res = plain_controller.rotsum_padded(res, 64);

    return res;
}

static Ctxt layer1(const Ctxt& in)
{
    bool timing = verbose > 1;
    double scale = 1.00;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt res1 = controller.convbn(in, 1, 1, scale, timing);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);

    scale = 0.52;

    res1 = controller.convbn(res1, 1, 2, scale, timing);
    res1 = controller.add(res1, controller.mult(in, scale));
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 1---" << std::endl;

    scale = 0.55;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn(res1, 2, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.36;

    res2 = controller.convbn(res2, 2, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 2---" << std::endl;

    scale = 0.63;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn(res2, 3, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.42;

    res3 = controller.convbn(res3, 3, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 3---" << std::endl;

    return res3;
}

static Ctxt layer2(const Ctxt& in)
{
    double scaleSx = 0.57;
    double scaleDx = 0.40;

    bool timing = verbose > 1;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    std::vector<Ctxt> res1sx =
        controller.convbn1632sx(boot_in, 4, 1, scaleSx, timing);
    std::vector<Ctxt> res1dx =
        controller.convbn1632dx(boot_in, 4, 1, scaleDx, timing);

    Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);

    controller.num_slots = 8192;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = controller.convbn2(fullpackSx, 4, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 1---" << std::endl;

    double scale = 0.76;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn2(res1, 5, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.37;

    res2 = controller.convbn2(res2, 5, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 2---" << std::endl;

    scale = 0.63;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn2(res2, 6, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.25;

    res3 = controller.convbn2(res3, 6, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 3---" << std::endl;

    return res3;
}

static Ctxt layer3(const Ctxt& in)
{
    double scaleSx = 0.63;
    double scaleDx = 0.40;

    bool timing = verbose > 1;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    std::vector<Ctxt> res1sx =
        controller.convbn3264sx(boot_in, 7, 1, scaleSx, timing);
    std::vector<Ctxt> res1dx =
        controller.convbn3264dx(boot_in, 7, 1, scaleDx, timing);

    Ctxt fullpackSx = controller.downsample256to64(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample256to64(res1dx[0], res1dx[1]);

    controller.num_slots = 4096;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = controller.convbn3(fullpackSx, 7, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 1---" << std::endl;

    double scale = 0.57;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn3(res1, 8, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.33;

    res2 = controller.convbn3(res2, 8, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 2---" << std::endl;

    scale = 0.69;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn3(res2, 9, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.10;

    res3 = controller.convbn3(res3, 9, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    res3 = controller.bootstrap(res3, timing);

    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 3---" << std::endl;

    return res3;
}

static Ctxt final_layer(const Ctxt& in)
{
    controller.num_slots = 4096;

    std::vector<double> fc_weight =
        utils::read_fc_weight(weights_dir + "/fc.bin");
    lowmem::Ptxt weight =
        controller.encode(fc_weight, in.depth(), controller.num_slots);

    Ctxt res = controller.rotsum(in, 64);
    res = controller.mult(res,
                          controller.mask_mod(64, res.depth(), 1.0 / 64.0));

    res = controller.repeat(res, 16);
    res = controller.mult(res, weight);
    res = controller.rotsum_padded(res, 64);

    if (verbose >= 0) {
        std::cout << "Decrypting the output..." << std::endl;
        controller.print(res, 10, "Output: ");
    }

    std::vector<double> clear_result = controller.decrypt_tovector(res, 10);
    auto max_it = std::max_element(clear_result.begin(), clear_result.end());
    int index_max = static_cast<int>(std::distance(clear_result.begin(), max_it));

    if (verbose >= 0) {
        std::cout << "The input image is classified as " << YELLOW_TEXT
                  << utils::get_class(index_max) << RESET_COLOR << std::endl;
        std::cout << "The index of max element is " << YELLOW_TEXT << index_max
                  << RESET_COLOR << std::endl;
    }

    if (plain) {
        std::string command =
            "python3 ../LowMemoryFHEResNet20/src/plain/script.py \"" +
            (data_dir + "/" + input_filename) + "\"";
        int return_sys = system(command.c_str());
        if (return_sys == 1) {
            std::cout << "Error launching plain script." << std::endl;
        }
    }

    return res;
}

static void execute_resnet20()
{
    if (verbose >= 0)
        std::cout << "Encrypted ResNet20 classification started." << std::endl;

    std::string input_path = data_dir + "/" + input_filename;
    if (verbose >= 0) {
        std::cout << "Encrypting and classifying " << GREEN_TEXT << input_path
                  << RESET_COLOR << std::endl;
    }

    std::vector<double> input_image = read_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Input image load failed. Check --data_dir and --input."
                  << std::endl;
        return;
    }

    plainsim::PlainVec plain_input = plain_controller.pad_input(input_image);
    if (plain_sim || compare_plain_sim) {
        plain_controller.weights_dir = weights_dir;
        plain_controller.num_slots = controller.num_slots;
        plain_controller.relu_degree = controller.relu_degree;
        plain_controller.relu_div_scale = controller.plain_relu_div_scale;
    }

    plainsim::PlainVec plain_initial;
    plainsim::PlainVec plain_layer1;
    plainsim::PlainVec plain_layer2;
    plainsim::PlainVec plain_layer3;
    plainsim::PlainVec plain_final;

    if (plain_sim || compare_plain_sim) {
        plain_initial = initial_layer_plain(plain_input);
        plain_layer1 = layer1_plain(plain_initial);
        plain_layer2 = layer2_plain(plain_layer1);
        plain_layer3 = layer3_plain(plain_layer2);
        plain_final = final_layer_plain(plain_layer3);
    }

    if (plain_sim && !compare_plain_sim) {
        auto max_it = std::max_element(plain_final.begin(),
                                       plain_final.begin() + 10);
        int index_max = static_cast<int>(std::distance(plain_final.begin(),
                                                       max_it));
        std::cout << "PlainSim top-1: " << utils::get_class(index_max)
                  << " (" << index_max << ")" << std::endl;
        return;
    }
    if (compare_plain_sim) {
        std::cout << "PlainSim output:[";
        for (int i = 0; i < 10; i++) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << std::fixed << std::setprecision(3)
                      << plain_final[static_cast<size_t>(i)];
        }
        std::cout << "]" << std::endl;
        auto max_it = std::max_element(plain_final.begin(),
                                       plain_final.begin() + 10);
        int index_max = static_cast<int>(std::distance(plain_final.begin(),
                                                       max_it));
        std::cout << "PlainSim top-1: " << utils::get_class(index_max)
                  << " (" << index_max << ")" << std::endl;
    }

    Ctxt in = controller.encrypt(
        input_image,
        controller.circuit_depth - 4 -
            utils::get_relu_depth(controller.relu_degree));

    auto start = utils::start_time();

    controller.print_level_scale(in, "Input");
    Ctxt firstLayer = initial_layer(in);
    controller.print_level_scale(firstLayer, "Initial layer out");
    if (debug_plain) {
        debug_decrypt("Initial layer out", firstLayer, 8);
    }
    if (compare_plain_sim) {
        compare_plain_sim_layer("Initial layer", plain_initial, firstLayer,
                                plain_controller.num_slots);
    }

    auto startLayer = utils::start_time();
    controller.print_level_scale(firstLayer, "Layer 1 in");
    Ctxt resLayer1 = layer1(firstLayer);
    controller.print_level_scale(resLayer1, "Layer 1 out");
    if (debug_plain) {
        debug_decrypt("Layer 1 out", resLayer1, 8);
    }
    if (compare_plain_sim) {
        compare_plain_sim_layer("Layer 1", plain_layer1, resLayer1,
                                plain_controller.num_slots);
    }
    utils::print_duration(startLayer, "Layer 1 took:");

    startLayer = utils::start_time();
    controller.print_level_scale(resLayer1, "Layer 2 in");
    Ctxt resLayer2 =
        compare_plain_sim ? layer2_compare(resLayer1, plain_layer1)
                          : layer2(resLayer1);
    controller.print_level_scale(resLayer2, "Layer 2 out");
    if (debug_plain) {
        debug_decrypt("Layer 2 out", resLayer2, 8);
    }
    if (compare_plain_sim) {
        compare_plain_sim_layer("Layer 2", plain_layer2, resLayer2,
                                plain_controller.num_slots);
    }
    utils::print_duration(startLayer, "Layer 2 took:");

    startLayer = utils::start_time();
    controller.print_level_scale(resLayer2, "Layer 3 in");
    Ctxt resLayer3 = layer3(resLayer2);
    controller.print_level_scale(resLayer3, "Layer 3 out");
    if (debug_plain) {
        debug_decrypt("Layer 3 out", resLayer3, 8);
    }
    if (compare_plain_sim) {
        compare_plain_sim_layer("Layer 3", plain_layer3, resLayer3,
                                plain_controller.num_slots);
    }
    utils::print_duration(startLayer, "Layer 3 took:");

    controller.print_level_scale(resLayer3, "Final layer in");
    Ctxt finalRes = final_layer(resLayer3);
    controller.print_level_scale(finalRes, "Final layer out");
    if (debug_plain) {
        debug_decrypt("Final layer out", finalRes, 10);
    }
    if (compare_plain_sim) {
        compare_plain_sim_layer("Final layer", plain_final, finalRes,
                                plain_controller.num_slots);
    }

    utils::print_duration_yellow(start,
                                 "The evaluation of the whole circuit took: ");

    std::cout << "Total mul: " << controller.mul_count
              << " rot: " << controller.rot_count
              << " rescale: " << controller.rescale_count
              << " boot: " << controller.boot_count << std::endl;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(1);

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

    HEConfig cfg;
    cfg.relu_degree = controller.relu_degree;
    controller.weights_dir = weights_dir;
    controller.initialize(cfg);
    controller.debug_cuda = debug_cuda;
    controller.debug_encode = debug_encode;

    execute_resnet20();

    return 0;
}
