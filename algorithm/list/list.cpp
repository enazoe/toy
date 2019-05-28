#include <iostream>

struct Node
{
	Node(int n_value)
	{
		_n_value = n_value;
		_p_next = NULL;
	}
	int _n_value;
	Node *_p_next;
};


class SingleList
{
	public:
		SingleList()
		{
			Node *p_temp = new Node(-1);
			_p_head = p_temp;
		}
		
		~SingleList()
		{
			Node *temp = _p_head -> _p_next;
			Node *p = NULL;
			while(temp!=NULL)
			{
				p = temp;
				temp = temp ->_p_next;
				delete p;
			}
			delete _p_head;
		}

		void dump()
		{
			Node *temp = _p_head -> _p_next;
			Node *p = NULL;
			while(temp!=NULL)
			{
				p = temp;
				temp = temp ->_p_next;
				free(p);
			}
		}
		void disp()
		{
			Node *temp =  _p_head->_p_next;
			while(temp !=NULL)
			{
				std::cout<<temp->_n_value<<std::endl;
				temp = temp->_p_next;
			}
		}
		void push_front(const int n_value_)
		{
			Node *p_temp = new Node(n_value_);
			p_temp->_p_next = _p_head->_p_next;
			_p_head-> _p_next = p_temp;
		}

		void push_back(const int n_value_)
		{
			Node *temp = _p_head -> _p_next;
			while(temp->_p_next != NULL)
			{
				temp = temp -> _p_next;
			}
			temp -> _p_next = new Node(n_value_);
			temp -> _p_next -> _p_next = NULL;
		}
	private:
		
		Node *_p_head;
};



int main()
{


	std::cout<<"test"<<std::endl;
	SingleList list;
	list.push_front(1);
	list.push_front(2);
	list.disp();
	list.dump();
	std::cout<<"test"<<std::endl;
	list.push_front(6);
	std::cout<<"test"<<std::endl;
	list.disp();
	std::cout<<"test"<<std::endl;
	list.push_back(7);
	std::cout<<"test"<<std::endl;
	list.disp();
	std::cin.get();
	return 0;
	
}
