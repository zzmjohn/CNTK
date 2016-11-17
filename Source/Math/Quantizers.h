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
    virtual void Dequantize(const ArrayRef<RawType>& input, ArrayRef<RawType>& output) = 0;


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

template <class RawType>
class QuantizedBlockMultiplier
{
private:
    shared_ptr<QuantizerBase<RawType, short>> m_quantizerA;
    shared_ptr<ArrayRef<short>> m_quantizedA;
    shared_ptr<QuantizerBase<RawType, short>> m_quantizerB;
    shared_ptr<ArrayRef<short>> m_quantizedB;
    shared_ptr<ArrayRef<RawType>> m_C;
    bool m_isAConstant;
    bool m_isBConstant;
    short *m_matA, *m_matB;
    int32_t* m_matC;
    size_t m_m, m_n, m_k;

public: 
    QuantizedBlockMultiplier(shared_ptr<QuantizerBase<RawType, short>> quantizerA, bool isAConstant, shared_ptr<QuantizerBase<RawType, short>> quantizerB, bool isBConstant) :
        m_quantizerA(quantizerA), m_quantizerB(quantizerB), m_isAConstant(isAConstant), m_isBConstant(isBConstant), m_quantizedA(nullptr), m_quantizedB(nullptr), m_C(nullptr)
    {
        if (isAConstant && isBConstant)
            LogicError("Quantized multiplication is applied to two constant matrices -- it is highly inefficient. Better approach is to replace the operation with the resulting matrix.");
    };

    ~QuantizedBlockMultiplier()
    {
        if (m_quantizedA)
            delete[] m_matA;

        if (m_quantizedB)
            delete[] m_matB;
    }

    // A[m,k]*B[k,n] = C[m,n]
    void Multiply(int m, int n, int k, RawType* A, RawType* B, RawType* C)
    {
        int mn = m*n;
        int nk = n*k;
        int mk = m*k;

        if (!m_quantizedA && !m_quantizedB && !m_C)
        {
            m_matA = new short[mk];
            m_m = m;
            m_n = n;
            m_k = k;
            m_quantizedA = shared_ptr<ArrayRef<short>>(new ArrayRef<short>(m_matA, mk));
            
            if (m_isAConstant)
                m_quantizerA->Quantize(ArrayRef<RawType>(A, mk), *m_quantizedA);

            m_matB = new short[nk];
            m_quantizedB = shared_ptr<ArrayRef<short>>(new ArrayRef<short>(m_matB, nk));

            if (m_isBConstant)
                m_quantizerB->Quantize(ArrayRef<RawType>(B, nk), *m_quantizedB);

            m_C = shared_ptr<ArrayRef<RawType>>(new ArrayRef<RawType>(C, mn));
        }

        // Quantize
        if (!m_isAConstant)
            m_quantizerA->Quantize(ArrayRef<RawType>(A, mk), *m_quantizedA);
        
        if (!m_isBConstant)
            m_quantizerB->Quantize(ArrayRef<RawType>(B, nk), *m_quantizedB);

        // Do multiply
        // Naive inefficient product, just for demonstation
        // TODO: replace with IPG multiplier
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
            {
                int dotProduct=0;
                for (size_t l = 0; l < k; l++)
                {
                    // CNTK is using column-major storage
                    dotProduct += (*m_quantizedA)[i + l*m] * (*m_quantizedB)[l + n*j];
                }
                (*m_C)[i + j*m] = (float)dotProduct;
            }

        // De-quantize
        m_quantizerB->Dequantize(*m_C, *m_C);
        m_quantizerA->Dequantize(*m_C, *m_C);
    }
};

}}}