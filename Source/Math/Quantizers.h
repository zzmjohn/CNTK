//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// RawType - input type to the quantizer. Currently CNTK supports float or double as RawType.
// QuantizedType - output type of the quantizer
template <class RawType, class QuantizedType>
class QuantizerBase 
{
public:
    QuantizerBase()
    {
        rangeMax = std::numeric_limits<QuantizedType>::max();
    }
    virtual void Quantize(const ArrayRef<RawType>& input, ArrayRef<QuantizedType>& output) = 0;
    virtual void Dequantize(const ArrayRef<RawType>& input, ArrayRef<RawType>& output) = 0;


protected:
    QuantizedType rangeMax;
};

// Symmetric quantizer. 
// Quantization is achieved by 
//    1. Finding the absolute max of values to be quantized.
//    2. Adjusting the max with bit shifting specified with the bitShift parameter (see comment at the declaration of the parameter)
//    3. Scaling all values in the collection to be within the symmetric range of the signed integer (QuantizedType)
template <class RawType, class QuantizedType>
class SymmetricQuantizer : public QuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizeFactor;
    RawType m_inverseQuantizerFactor;
    RawType m_absMax;

    // Decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    // Higher bitShift will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    // For quantization with shorts, recommended value of bitShift is from 1 to 3.
    size_t m_bitShift; 
public:
    // elements - collection to be quantized
    // bitShift - see comment above
    SymmetricQuantizer(size_t bitShift) :m_bitShift(bitShift)
    {
    }

    // Perform quantization of the input collection, put result into pre-allocated output collection
    virtual void Quantize(const ArrayRef<RawType>& input, ArrayRef<QuantizedType>& output)
    {
        if (input.size() == 0) return;

        Initialize(FindAbsMax(input));
        assert(input.size() == output.size());

        for (size_t i = 0; i < input.size(); i++)
        {
#ifdef _DEBUG
            assert(abs(input[i]) <= m_absMax);
#endif
            output[i] = (QuantizedType) round(input[i] * m_quantizeFactor);
        }
    }

    // Accept quantized collection as input, put de-quantization result into pre-allocated output collection.
    virtual void Dequantize(const ArrayRef<RawType>& input, ArrayRef<RawType>& output)
    {
        assert(input.size() == output.size());

        for (size_t i = 0; i < input.size(); i++)
        {
            output[i] = input[i] * m_inverseQuantizerFactor;
        }
    }

private: 
    // Find absolute maximum value
    RawType FindAbsMax(const ArrayRef<RawType>& arrayRef)
    {
        auto minMaxPair = std::minmax_element(arrayRef.begin(), arrayRef.end());

        return std::max(arrayRef[minMaxPair.second - arrayRef.begin()], std::abs(arrayRef[minMaxPair.first - arrayRef.begin()]));
    }

    void Initialize(RawType absoluteMax)
    {
        RawType shiftedMax = absoluteMax * (1 << m_bitShift);
        if (shiftedMax == 0)
        {
            LogicError("The absolute max element in the sequence to be quantized is 0.");
        }
        m_absMax = absoluteMax;
        m_quantizeFactor = this->rangeMax / shiftedMax;
        m_inverseQuantizerFactor = 1 / m_quantizeFactor;
    }
};

}}}