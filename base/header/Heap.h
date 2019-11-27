//---------//
// Heap.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _BoraHeap_h_
#define _BoraHeap_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <class T, class S>
struct HeapNode
{
	T data;
	S value;

	HeapNode( const T& inData, const S& inValue )
	: data(inData), value(inValue)
	{}
};

template <class T, class S>
class MinHeap
{
	protected:

		struct Compare
		{
			bool operator()( const HeapNode<T,S>& a, const HeapNode<T,S>& b )
			{
				return ( a.value > b.value );
			}
		};

		std::priority_queue< class HeapNode<T,S>, std::vector<class HeapNode<T,S> >, class MinHeap<T,S>::Compare > _data;

	public:

		MinHeap() {}

		void push( const HeapNode<T,S>& n ) { _data.push(n); }

		const class HeapNode<T,S>& top() { return _data.top(); }

		void pop() { return _data.pop(); }

		bool empty() { return _data.empty(); }

		int size() const { return (int)_data.size(); }

		void clear() { _data = std::priority_queue< class HeapNode<T,S>, std::vector<class HeapNode<T,S> >, class MinHeap<T,S>::Compare >(); }
};

template <class T, class S>
class MaxHeap
{
	protected:

		struct Compare
		{
			bool operator()( const HeapNode<T,S>& a, const HeapNode<T,S>& b )
			{
				return ( a.value < b.value );
			}
		};

		std::priority_queue< class HeapNode<T,S>, std::vector<class HeapNode<T,S> >, class MaxHeap<T,S>::Compare > _data;

	public:

		MaxHeap() {}

		void push( const HeapNode<T,S>& n ) { _data.push(n); }

		const class HeapNode<T,S>& top() { return _data.top(); }

		void pop() { return _data.pop(); }

		bool empty() { return _data.empty(); }

		int size() const { return (int)_data.size(); }

		void clear() { _data = std::priority_queue< class HeapNode<T,S>, std::vector<class HeapNode<T,S> >, class MaxHeap<T,S>::Compare >(); }
};

////////////////
// data types //

typedef HeapNode<Vec3i,float> HEAPNODE;

BORA_NAMESPACE_END

#endif

