//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "Basics.h"
#undef max
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
    virtual void Dequantize(const ArrayRef<QuantizedType>& input, ArrayRef<RawType>& output) = 0;

protected:
    QuantizedType rangeMax;
};

// Symmetric quantizer. 
// Quantization is achieved by 
//    1. Finding the absolute max of values to be quantized.
//    2. Adjusting the absolute max with extraBits parameter.
//    3. Scaling all values in the collection to be within the symmetric range of the QuantizedType
template <class RawType, class QuantizedType>
class SymmetricQuantizer : public QuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizeFactor;
    RawType m_inverseQuantizerFactor;
    RawType m_absMax;
    size_t m_extraBits;
public:
    // elements - collection to be quantized
    // extraBits decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    //     Higher extraBits will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    //     For quantization with shorts, recommended value of extraBits is 1-3.
    // This constructor accepts the collection of RawType to initialize internal quantizer
    // and then apply this quantizer to collections with similar range as the one it was initialized with.
    SymmetricQuantizer(size_t extraBits) :m_extraBits(extraBits)
    {
    }

    // Perform quantization of the input collection, put result into pre-allocated output collection
    virtual void Quantize(const ArrayRef<RawType>& input, ArrayRef<QuantizedType>& output)
    {
        Initialize(FindAbsMax(input), m_extraBits);
        assert(input.size() == output.size());

        for (size_t i = 0; i < input.size(); i++)
        {
#ifdef _DEBUG
            assert(abs(input[i]) <= m_absMax);
#endif
            output[i] = (QuantizedType) round((input[i] * m_quantizeFactor));
        }
    }

    // Accept quantized collection as input, put de-quantization result into pre-allocated output collection.
    virtual void Dequantize(const ArrayRef<QuantizedType>& input, ArrayRef<RawType>& output)
    {
        assert(input.size() == output.size());

        for (size_t i = 0; i < input.size(); i++)
        {
            output[i] = (RawType)(input[i] * m_inverseQuantizerFactor);
        }
    }

private: 
    // Find absolute maximum value
    RawType FindAbsMax(const ArrayRef<RawType>& arrayRef)
    {
        RawType maxElem = *std::max_element(arrayRef.begin(), arrayRef.end());
        RawType minElem = *std::min_element(arrayRef.begin(), arrayRef.end());

        return std::max(maxElem, std::abs(minElem));
    }

    void Initialize(RawType absoluteMax, size_t extraBits)
    {
        RawType shiftedMax = absoluteMax * (1 << extraBits);
        if (shiftedMax == 0)
        {
            LogicError("The absolute max element in the sequence to be quantized is 0.");
        }
        m_absMax = absoluteMax;
        m_quantizeFactor = this->rangeMax / shiftedMax;
        m_inverseQuantizerFactor = 1 / m_quantizeFactor;
    }
};

template <class RawType, class QuantizedType>
class QuantizedBlockMultiplier
{
private:
    shared_ptr<QuantizerBase<RawType, QuantizedType>> m_quantizerA;
    shared_ptr<ArrayRef<QuantizedType>> m_quantizedA;
    shared_ptr<QuantizerBase<RawType, QuantizedType>> m_quantizerB;
    shared_ptr<ArrayRef<QuantizedType>> m_quantizedB;
    bool m_isAConstant;

public: 
    QuantizedBlockMultiplier(shared_ptr<ArrayRef<QuantizedType>> quantizedA, shared_ptr<QuantizerBase<RawType, QuantizedType>> quantizerA, shared_ptr<ArrayRef<QuantizedType>> quantizedB, shared_ptr<QuantizerBase<RawType, QuantizedType>> quantizerB, bool AConstant) :
        m_quantizedA(quantizedA), m_quantizerA(quantizerA), m_quantizedB(quantizedB), m_quantizerB(quantizerB), m_AConstant(isAConstant)
    {
    }

    void Multiply(RawType* A, RawType* B, int m, int n, int k, RawType* C)
    {
    }

};

}}}