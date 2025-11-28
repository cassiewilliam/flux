//===- gen_ag_gemm.cc -------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "./generator_utils.h"
#include "flux/flux.h"

namespace bytedance::flux::generator {
using namespace cute;

struct GemmV3AGKernel_Space
{

    static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
        cute::make_tuple(make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}, _FP32{}),
                         make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _FP16{}, _FP32{})),
        cute::make_tuple(_Sm90{}),
        cute::make_tuple(_H800{}),
        cute::make_tuple(_AGKernel{}),
        cute::make_tuple(_RCR{}),
        cute::make_tuple(_GemmV3{}),
        cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

    static constexpr auto AllGemmHParams_FP8 = tuple_cat(
        make_space_gemm_hparams(
            cute::make_tuple(make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _PingPong{}),
                             make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _PingPong{})),
            cute::make_tuple(Auto{}),
            cute::make_tuple(Shape<_64, _128, _128>{}),
            cute::make_tuple(_GemmDefault{}),
            cute::make_tuple(cute::_8{}),
            cute::make_tuple(_RasterAlongN{})),
        make_space_gemm_hparams(
            cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{})),
            cute::make_tuple(Auto{}),
            cute::make_tuple(Shape<_128, _256, _128>{}),
            cute::make_tuple(_GemmStreamK{}),
            cute::make_tuple(cute::_4{}),
            cute::make_tuple(_RasterHeuristic{})),
        make_space_gemm_hparams(
            cute::make_tuple(make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _Cooperative{})),
            cute::make_tuple(Auto{}),
            cute::make_tuple(Shape<_128, _128, _128>{}),
            cute::make_tuple(_GemmDefault{}),
            cute::make_tuple(cute::_3{}),
            cute::make_tuple(_RasterAlongN{})));

    static auto get_space()
    {
        return merge_gen_space({
            build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
        });
    }
};
}   // namespace bytedance::flux::generator

int main(int argc, char const** args)
{
    using namespace bytedance::flux::generator;
    Options options;
    options.parse(argc, args);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    std::cout << "Running ag_gemm generator...\n";
    return main_template(options,
                         {
                             cute::make_tuple(GemmV3AGKernel_Space::get_space(),
                                              std::string("ag_gemm/gemm_v3_ag_kernel.hpp"),
                                              std::string("GemmV3AGKernel")),
                         });
}
