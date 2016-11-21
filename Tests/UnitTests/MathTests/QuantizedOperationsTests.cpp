//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/QuantizedOperations.h"
#include "../../../Source/Math/Helpers.h"

using namespace Microsoft::MSR::CNTK;
namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(QuantizedOperationsUnitTests)

BOOST_FIXTURE_TEST_CASE(MultiplyIntToShort, RandomSeedFixture)
{
    // A[m,k]*B[k,n] = C[m,n]
    int m = 5, n = 4, k = 3;
    std::vector<int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}; 
    std::vector<int> B = {16,17,18,19,20,21,22,23,24,25,26,27}; 
    std::vector<int> C_expected = { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35 };
    std::vector<int> C;
    C.reserve(m*n);

    shared_ptr<QuantizerBase<int, short>> quantA(new SymmetricQuantizer<int, short>(1));
    shared_ptr<QuantizerBase<int, short>> quantB(new SymmetricQuantizer<int, short>(2));

    // A - is constant; B - is not
    QuantizedMultiplier<int> mult(quantA, true, quantB, false);

    // First pass
    mult.Multiply(m, n, k, A.data(), B.data(), C.data());

    for (size_t i = 0; i < m*n; i++)
        BOOST_CHECK_EQUAL(C[i], C_expected[i]);

    // Second pass, the same matrices
    mult.Multiply(m, n, k, A.data(), B.data(), C.data());

    for (size_t i = 0; i < m*n; i++)
        BOOST_CHECK_EQUAL(C[i], C_expected[i]);

    // Third pass with updated B (size and values)
    int n_upd = 5;
    std::vector<int> B_upd = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    std::vector<int> C_expected_upd = { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 };
    std::vector<int> C_upd;
    C_upd.reserve(m*n_upd);
    mult.Multiply(m, n_upd, k, A.data(), B_upd.data(), C_upd.data());
    for (size_t i = 0; i < m*n_upd; i++)
        BOOST_CHECK_EQUAL(C_upd[i], C_expected_upd[i]);
}


BOOST_AUTO_TEST_SUITE_END()

} } } }