//
// Created by Chi Zhang on 5/12/22.
//

#ifndef TPDS22_CODE_SAFE_QUEUE_H
#define TPDS22_CODE_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

// A threadsafe-queue.
template<class T>
class SafeQueue {
public:
    explicit SafeQueue(int max_size) : max_size(max_size) {}

    ~SafeQueue() = default;

    int size() {
        std::lock_guard<std::mutex> lock(m);
        int size = q.size();
        return size;
    }

    // Add an element to the queue.
    void enqueue(T t) {
        std::unique_lock<std::mutex> lock(m);
        while ((int) q.size() == max_size) {
            enqueue_c.wait(lock);
        }
        q.push(t);
        deque_c.notify_one();
    }

    bool try_enqueue(T t) {
        bool success = false;
        std::unique_lock<std::mutex> lock(m);
        if ((int) q.size() != max_size) {
            q.push(t);
            deque_c.notify_one();
            success = true;
        }
        return success;
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue() {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            // release lock as long as the wait and reaquire it afterwards.
            deque_c.wait(lock);
        }
        T val = q.front();
        q.pop();
        enqueue_c.notify_one();
        return val;
    }

    std::optional<T> try_dequeue() {
        std::unique_lock<std::mutex> lock(m);
        if (q.empty()) {
            return {};
        } else {
            T val = q.front();
            q.pop();
            enqueue_c.notify_one();
            return val;
        }
    }

private:
    int max_size;
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable deque_c;
    std::condition_variable enqueue_c;
};


#endif //TPDS22_CODE_SAFE_QUEUE_H
