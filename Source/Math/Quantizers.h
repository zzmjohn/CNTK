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
//    2. Adjusting the absolute max with bitSmoothing parameter.
//    3. Scaling all values in the collection to be within the symmetric range of the QuantizedType
template <class RawType, class QuantizedType>
class SymmetricQuantizer : public QuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizeFactor;
    RawType m_inverseQuantizerFactor;
    RawType m_absMax;
    size_t m_bitSmoothing;
public:
    // elements - collection to be quantized
    // bitSmoothing decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    //     Higher bitSmoothing will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    //     For quantization with shorts, recommended value of bitSmoothing is 1-3.
    SymmetricQuantizer(size_t bitSmoothing) :m_bitSmoothing(bitSmoothing)
    {
    }

    // Perform quantization of the input collection, put result into pre-allocated output collection
    virtual void Quantize(const ArrayRef<RawType>& input, ArrayRef<QuantizedType>& output)
    {
        Initialize(FindAbsMax(input), m_bitSmoothing);
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

    void Initialize(RawType absoluteMax, size_t bitSmoothing)
    {
        RawType shiftedMax = absoluteMax * (1 << bitSmoothing);
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
class QuantizedMultiplier
{
private:
    shared_ptr<QuantizerBase<RawType, short>> m_pQuantizerA;
    shared_ptr<ArrayRef<short>> m_pQuantizedA;
    shared_ptr<QuantizerBase<RawType, short>> m_pQuantizerB;
    shared_ptr<ArrayRef<short>> m_pQuantizedB;
    shared_ptr<ArrayRef<RawType>> m_pC;
    bool m_isAConstant;
    bool m_isBConstant;
    short *m_pMatA, *m_pMatB;
    size_t m_m, m_n, m_k;

public: 
    QuantizedMultiplier(shared_ptr<QuantizerBase<RawType, short>> pQuantizerA, bool isAConstant, shared_ptr<QuantizerBase<RawType, short>> pQuantizerB, bool isBConstant) :
        m_pQuantizerA(pQuantizerA), m_pQuantizerB(pQuantizerB), m_isAConstant(isAConstant), m_isBConstant(isBConstant), m_pQuantizedA(nullptr), m_pQuantizedB(nullptr), m_pC(nullptr)
    {
        if (isAConstant && isBConstant)
            LogicError("Quantized multiplication is applied to two constant matrices -- it is highly inefficient. Better approach is to replace the operation with the resulting matrix.");
    };

    ~QuantizedMultiplier()
    {
        if (m_pQuantizedA)
            delete[] m_pMatA;

        if (m_pQuantizedB)
            delete[] m_pMatB;
    }

    // A[m,k]*B[k,n] = C[m,n]
    void Multiply(int m, int n, int k, RawType* A, RawType* B, RawType* C)
    {
        int mn = m*n;
        int nk = n*k;
        int mk = m*k;

        if (!m_pQuantizedA && !m_pQuantizedB && !m_pC)
        {
            m_pMatA = new short[mk];
            m_m = m;
            m_n = n;
            m_k = k;
            m_pQuantizedA = shared_ptr<ArrayRef<short>>(new ArrayRef<short>(m_pMatA, mk));
            
            if (m_isAConstant)
                m_pQuantizerA->Quantize(ArrayRef<RawType>(A, mk), *m_pQuantizedA);

            m_pMatB = new short[nk];
            m_pQuantizedB = shared_ptr<ArrayRef<short>>(new ArrayRef<short>(m_pMatB, nk));

            if (m_isBConstant)
                m_pQuantizerB->Quantize(ArrayRef<RawType>(B, nk), *m_pQuantizedB);

            m_pC = shared_ptr<ArrayRef<RawType>>(new ArrayRef<RawType>(C, mn));
        }

        // Quantize
        if (!m_isAConstant)
            m_pQuantizerA->Quantize(ArrayRef<RawType>(A, mk), *m_pQuantizedA);
        
        if (!m_isBConstant)
            m_pQuantizerB->Quantize(ArrayRef<RawType>(B, nk), *m_pQuantizedB);

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
                    dotProduct += (*m_pQuantizedA)[i + l*m] * (*m_pQuantizedB)[l + n*j];
                }
                (*m_pC)[i + j*m] = (float)dotProduct;
            }

        // De-quantize
        m_pQuantizerB->Dequantize(*m_pC, *m_pC);
        m_pQuantizerA->Dequantize(*m_pC, *m_pC);
    }
};

}}}