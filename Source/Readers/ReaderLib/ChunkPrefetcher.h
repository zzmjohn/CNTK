//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include <queue>
#include <condition_variable>
#include "DataDeserializer.h"
#include <set>
#include <thread>
#include <atomic>

namespace Microsoft { namespace MSR { namespace CNTK {
/*
template<typename Element>
class blocking_queue
{
    typedef std::unique_lock<std::mutex> guard_type;

    std::queue<Element> m_queue;
    

public:
    void push(const Element& elem)
    {
        {
            guard_type guard(m_lock);
            m_queue.push(elem);
        }

        m_notifier.notify_one();
    }

    bool empty() const
    {
        guard_type guard(m_lock);
        return m_queue.empty();
    }

    bool try_pop(Element& result)
    {
        guard_type guard(m_lock);
        if (m_queue.empty())
        {
            return false;
        }

        result = m_queue.front();
        m_queue.pop();
        return true;
    }

    void pop(Element& result)
    {
        guard_type guard(m_lock);
        while (m_queue.empty())
        {
            m_notifier.wait(guard);
        }

        result = m_queue.front();
        m_queue.pop();
    }

    void clear()
    {
        guard_type guard(m_lock);
        while (!m_queue.empty())
        {
            m_queue.pop();
        }
    }
};
*/

class ChunkPrefetcher
{
    typedef void (*DeleteThreadType)(std::thread* ptr);
    static void DeleteThread(std::thread* ptr)
    {
        ptr->join();
        delete ptr;
    }

public:
    ChunkPrefetcher(IDataDeserializerPtr deserializer) :
        m_deserializer(deserializer),
        m_prefetcher(nullptr, DeleteThread)
    {}

    void Start()
    {
        m_stopFlag = false;
        m_prefetcher = std::unique_ptr<std::thread, DeleteThreadType>(
            new std::thread(&ChunkPrefetcher::Process, this), DeleteThread);
    }

    void Stop()
    {
        m_stopFlag = true;
        m_notifier.notify_all();
        m_prefetcher = nullptr;
    }

    void Prefetch(const std::vector<ChunkIdType>& chunks)
    {
        std::vector<ChunkIdType> newChunks;
        for (auto chunkId : chunks)
        {
            if (m_chunkIds.find(chunkId) == m_chunkIds.end())
            {
                m_chunkIds.insert(chunkId);
                newChunks.push_back(chunkId);
            }
        }

        guard_type guard(m_lock);
        m_toBePrefetched.insert(m_toBePrefetched.end(), newChunks.begin(), newChunks.end());
        m_notifier.notify_all();
    }

    ChunkPtr GetPrefetchedChunk(ChunkIdType chunkId)
    {
        if (m_chunkIds.find(chunkId) == m_chunkIds.end())
        {
            RuntimeError("Asked for not prefetched chunk.");
        }

        ChunkPtr result;
        {
            guard_type guard(m_lock);
            while (m_chunks.find(chunkId) == m_chunks.end())
            {
                m_notifier.wait(guard);
            }

            result = m_chunks[chunkId];
            m_chunks.erase(chunkId);
        }
        m_chunkIds.erase(chunkId);
        return result;
    }

    void Clear()
    {
        guard_type guard(m_lock);
        m_toBePrefetched.clear();
    }

private:

    void Process()
    {
        std::vector<ChunkIdType> ids;
        while (!this->m_stopFlag)
        {
            {
                guard_type guard(this->m_lock);
                while (this->m_toBePrefetched.empty() && !this->m_stopFlag)
                {
                    this->m_notifier.wait(guard);
                }

                ids.assign(this->m_toBePrefetched.begin(), this->m_toBePrefetched.end());
                this->m_toBePrefetched.clear();
            }

            for (auto id : ids)
            {
                ChunkPtr chunk = this->m_deserializer->GetChunk(id);
                guard_type guard(this->m_lock);
                this->m_chunks[id] = chunk;
            }

            m_notifier.notify_all();
        }
    }

    std::set<ChunkIdType> m_chunkIds;
    std::vector<ChunkIdType> m_toBePrefetched;
    std::map<ChunkIdType, ChunkPtr> m_chunks;
    std::unique_ptr<std::thread, DeleteThreadType> m_prefetcher;
    std::atomic<bool> m_stopFlag;

    IDataDeserializerPtr m_deserializer;

    
    typedef std::unique_lock<std::mutex> guard_type;
    mutable std::mutex m_lock;
    std::condition_variable m_notifier;

    DISABLE_COPY_AND_MOVE(ChunkPrefetcher);
};

}}}
