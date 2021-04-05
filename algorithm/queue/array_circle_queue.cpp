#include <iostream>

template<class T>
class ArrayCircleQueue
{

public:
	ArrayCircleQueue(int n)
	{
        _p_queue = new T[n];
        _size = n;
	}

	~ArrayCircleQueue()
	{
        delete [] _p_queue;
	}

	bool enqueue(const T &value_)
	{
        if((_tail+1)%_size == _head)
        {
            return false;
        }
        _p_queue[_tail] = value_;
        _tail = (_tail+1)%_size;
        return true;
	}

	T dequeue()
	{
        if(_head == _tail)
        {
            return NULL;
        }
        T temp = _p_queue[_head];
        
        _head = (_head+1)%_size;
        
        return temp;
        
	}
    
    void disp()
    {
        for (int i=_head;i%_size!=_tail; ++i)
        {
            std::cout<<_p_queue[i]<<std::endl;
        }
    }
private:

    int _head=0;
    int _tail=0;
    int _size =0;
    T*  _p_queue;

};

int main()
{
    ArrayCircleQueue<int> m_queue(10);
    m_queue.enqueue(1);
    m_queue.enqueue(2);
    m_queue.enqueue(3);
    m_queue.enqueue(4);
    m_queue.enqueue(5);
    m_queue.enqueue(6);
    m_queue.enqueue(7);
    m_queue.enqueue(8);
    m_queue.enqueue(9);
    m_queue.enqueue(10);
    m_queue.enqueue(11);
    m_queue.disp();
    std::cout<<"deque:"<<m_queue.dequeue()<<std::endl;
    m_queue.disp();
}
