#include "core/device.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

template <class Variant>
std::size_t expected_family_bytes(unsigned context, unsigned verify, unsigned neural) {
    constexpr std::size_t mib = 1ULL << 20;
    std::map<std::uint32_t, std::size_t> classes;
    const auto profiles = Variant::mtp_graph_profiles(context, verify);
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        const auto& profile = profiles[i];
        // Equal-width calls can share the Variant's stock topology. Unequal
        // target/next-draft widths require one executable per frontier range.
        const auto identity =
            verify == neural ? profile.topology_class : static_cast<std::uint32_t>(i);
        const auto visible = std::min<std::uint64_t>(
            context, static_cast<std::uint64_t>(profile.max) + verify + neural);
        classes[identity] = std::max(classes[identity], (visible <= 4096 ? 12U : 82U) * mib);
    }
    std::size_t total = 0;
    for (const auto& [identity, bytes] : classes) {
        (void)identity;
        total += bytes;
    }
    return total;
}

template <class Package, class Variant>
int verify_family(ninfer::DeviceContext& device, typename Package::WeightsProfile weights) {
    int failures   = 0;
    unsigned cases = 0;
    for (const unsigned context : {128U, 4090U, 4096U, 8192U, 32768U, 262144U}) {
        for (const unsigned neural : {1U, 2U, 3U, 4U, 5U}) {
            for (const unsigned ngram : {0U, 1U, 2U, 3U, 4U, 5U, 15U, 31U, 32U, 33U, 47U, 63U}) {
                std::array<std::size_t, 2> bytes{};
                for (unsigned graphs = 0; graphs < 2; ++graphs) {
                    ninfer::EngineOptions options;
                    options.max_context     = context;
                    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(context);
                    options.max_concurrency = 1;
                    options.prefill_chunk   = 1024;
                    options.enable_vision   = false;
                    options.kv_cache        = ninfer::KvCacheStorage::Nvfp4Group16;
                    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
                    options.speculative.draft_tokens         = neural;
                    options.speculative.ngram_draft_tokens   = ngram;
                    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
                    options.context_cache.device_state_slots = 1;
                    options.use_cuda_graph                   = graphs != 0;
                    auto planner     = Package::make_sequence_planner(device, options, weights);
                    const auto pages = planner.capacity_curve().minimum_main_page_groups;
                    bytes[graphs] = std::move(planner).finalize(pages).device_reservation_bytes();
                }
                const auto expected =
                    expected_family_bytes<Variant>(context, neural, neural) +
                    (ngram ? expected_family_bytes<Variant>(context, ngram, neural) : 0);
                ++cases;
                if (bytes[1] < bytes[0] || bytes[1] - bytes[0] != expected) {
                    ++failures;
                    std::cerr << Package::model_id << " graph allowance context=" << context
                              << " neural=" << neural << " ngram=" << ngram
                              << " actual=" << bytes[1] - bytes[0] << " expected=" << expected
                              << '\n';
                }
            }
        }
    }
    std::cout << Package::model_id << " graph planning: " << cases << " cases, " << failures
              << " failures\n";
    return failures;
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        using Dense  = ninfer::targets::qwen3_6_27b::Package;
        using Moe    = ninfer::targets::qwen3_6_35b_a3b::Package;
        int failures = verify_family<Dense, ninfer::targets::qwen3_6_27b::detail::Variant>(
            device, Dense::WeightsProfile::Qwen38Nvfp4);
        failures += verify_family<Moe, ninfer::targets::qwen3_6_35b_a3b::detail::Variant>(
            device, Moe::WeightsProfile::GroupwiseInt);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
