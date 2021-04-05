//
//  main.cpp
//  list
//
//  Created by 韩昭嵘 on 2019/5/29.
//  Copyright © 2019 韩昭嵘. All rights reserved.
//

#include <iostream>

struct Node
{

    Node(int n_value):
    _n_value(n_value),
    _p_next(NULL)
    {
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
            p = _p_head -> _p_next;
            _p_head->_p_next = p->_p_next;
            delete p;
            temp = _p_head->_p_next;
        }
        delete _p_head;
    }
    
    void dump()
    {
        Node *temp = _p_head -> _p_next;
        Node *p = NULL;
        while(temp!=NULL)
        {
            p = _p_head -> _p_next;
            _p_head->_p_next = p->_p_next;
            delete p;
            temp = _p_head->_p_next;
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
        Node *temp = _p_head;
        while(temp-> _p_next)
        {
            temp = temp -> _p_next;
        }
        temp-> _p_next= new Node(n_value_);
    }
    
    void reverse()
    {
        Node *p =_p_head->_p_next;
        Node *q = p->_p_next;
        Node *r;
        p->_p_next = NULL;
        while(q)
        {
            r= q->_p_next;
            q->_p_next =p;
            p=q;
            q=r;
        }
        _p_head->_p_next=p;
    }
    
    
    bool is_has_circle()
    {
        Node *slow = _p_head;
        Node *fast = _p_head;
        while(fast && fast ->_p_next)
        {
            slow = slow->_p_next;
            fast = fast->_p_next->_p_next;

            if (slow == fast)
            {
                return true;
            }
        }
        return false;
    }
    
private:
    
    Node *_p_head;
};



int main()
{
    while (true)
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
    list.push_front(1);
    list.push_back(2);
    list.push_back(3);
    list.disp();
    std::cout<<"reverse"<<std::endl;
    list.reverse();
    list.disp();
    list.dump();
    std::cout<<"push"<<std::endl;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.disp();
    if (list.is_has_circle())
    {
        std::cout<<"has circle"<<std::endl;
    }
    std::cin.get();
    }
    return 0;
    
}

