#include <iostream>

template<class T>
class ArrayStack
{
public:
	ArrayStack()
	{}

	ArrayStack(int count)
	{
		_array = new T[count];	
		_count = count;
		_top =0;
	}

	~ArrayStack()
	{}

	void  push(const T value_)
	{
		if(_top == _count)
		{
			std::cout<<"stack if full , so need to enlarge 2x!"<<std::endl;
			_count = 2* _count;
			T* temp = new T[_count];
			for(int i=0;i< _top;++i)
			{
				temp[i] = _array[i];
			}
			delete []_array;
			temp[_top] = value_;
			_top++;
			_array = temp;
		}
		else
		{
			_array[_top] = value_;
			_top++;
		}
	}

	T pop()
	{
		_top--;
		return _array[_top];
	}

	void disp()
	{
		for(int i=_top-1 ;i>-1;--i)
		{
			std::cout<<_array[i]<<std::endl;
		}
	}
private:
	
	T* _array;

	int _count= 0 ;

	int _top = 0;

};


int main()
{
	ArrayStack<int> m_stack(5);
	m_stack.push(1);
	m_stack.push(2);
	m_stack.push(3);
	m_stack.push(4);
	m_stack.push(5);
	m_stack.disp();
	m_stack.push(6);
	m_stack.disp();
	std::cout<<"pop"<<std::endl;
	std::cout<<m_stack.pop()<<std::endl;
	std::cout<<"pop"<<std::endl;
	m_stack.disp();
}
