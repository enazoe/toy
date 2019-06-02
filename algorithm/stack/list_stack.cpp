
#include <iostream>

template <class T>
class ListStack
{
public:	
	ListStack()
	{
		_head = new Node();
	}

	~ListStack()
	{
		Node *temp = _head -> next;
		Node *d = NULL;
		while(temp)
		{
			d = temp;
			temp = temp ->next;
			delete d;
		}
		delete _head;
		_head = NULL;
		_count = 0;
	}

	void disp()
	{
		Node *temp = _head->next;
		while(temp)
		{
			std::cout<<temp -> data <<std::endl;
			temp= temp->next;
		}
	}


	void push(const T &value_)
	{
		Node *temp = new Node();
		temp->data = value_;
		temp->next = _head->next;
		_head->next = temp;
		_count++;
	}
	
	T pop()
	{
		if(_count==0 || _head->next==NULL)
		{
			return NULL;
		}
		else
		{
			Node *temp = _head -> next;
			_head -> next = temp -> next;
			T data = temp ->data;
			delete temp;
			_count --;
			return data;
		}

	}
private:
	
	struct Node
	{
		T data;
		Node *next=NULL;
	};
	int _count=0;
	Node *_head;
};


int main()
{
	ListStack<int> m_list_stack;
	m_list_stack.push(1);
	m_list_stack.push(2);
	m_list_stack.push(3);
	m_list_stack.push(4);
	m_list_stack.disp();
	std::cout<< "pop:" << std::endl;
	std::cout<<m_list_stack.pop() <<std::endl;
	std::cout<< "pop:" << std::endl;
	m_list_stack.disp();
}
