//---------//
// Array.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.08.29                               //
//-------------------------------------------------------//

#ifndef _BoraArray_h_
#define _BoraArray_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

/////////////
// Caution ///////////////////////////
//                                  //
// case 1) b is the reference of a. //
// Array a( 10 );                   //
// Array  b( a );                   //
// Array  b = a;                    //
// Array& b = a;                    //
// Array b; b = a;                  //
//                                  //
// case 2) b is the copy of a.      //
// Array a( 10 );                   //
// Array b; b.copyFrom( a );        //
//                                  //
// case 3) b is the reference of a. //
// void Func( Array b ) {}          //
// Array a( 10 );                   //
// Func( a );                       //
//                                  //
//////////////////////////////////////

/// @brief A 1D array class.
/**
	This class implements a 1D array of various data types in Bora system.
*/
template<class TT>
class Array
{
    protected:

        MemorySpace _memorySpace     = kHost;     // 0: host, 1: device, 2: unified        
        TT*         _data            = (TT*)NULL; // the pointer of the data
        size_t      _size            = 0;         // the size of the array
        size_t      _capacity        = 0;         // the capacity of the array
        float       _increasingRatio = 2.f;       // the increasing ratio of the array
        bool        _referenceMode   = false;     // if true, the array is the reference of other array

    public:

        /// @brief The default constructor
        /**
            It creates a new empty instance which contains no elements.
        */
        BORA_UNIFIED
        Array();

        /// @brief The copy constructor
        /**
            It creates a new reference of the given array.
            @param[in] a The array object to copy from
        */
        BORA_UNIFIED
        Array( const Array<TT>& a );

        /// @brief The class constructor
        /**
            It creates a new array instance so that the instance has the given size in the specified memory space.
            It just calls Array::initialize().
            @param[in] n The size of the array to be set
            @param[in] memorySpace The memory space in which it lives
        */
        Array( const size_t n, const MemorySpace memorySpace=kHost );

        /// @brief The class destroyer
        /**
            It returns itself to the beginning state.
            It just calls Array::finalize(), but it does nothing for the reference mode.
        */
        BORA_UNIFIED
        virtual ~Array();

        /// @brief The function to set the instance
        /**
            It re-initializes the instance by the given arguments.
            @param[in] n The size of the array to be set
            @param[in] memorySpace The memory space in which it lives
        */
        void initialize( const size_t n, const MemorySpace memorySpace=kHost );

        /// @brief The function to reset the instance
        /**
            It returns itself to the beginning state.
        */
        void finalize();

        /// @brief The function to release the allocated memory
        /**
            It just sets the size of the array zero.
            It does not release the allocated memory, so the capacity does not changed.
        */
        void clear();

        /// @brief The function to reserve storage
        /**
            It makes its capacity be at least enough to contain n elements.
            If n is greater than the current capcity, it causes the array to re-allocate its storage so that its capacity is not smaller than n.
            In all other cases, the function call does not do anything, so the capacity is not affected.
            This function has no effect on the size and does not alter its elements.
            @param[in] n The new capacity
        */
        void reserve( const size_t n );

        /// @brief The function to resize the instance
        /**
            It resizes the array so that it contains n elements.
            This function does not alter its elements.
            It may cause a re-allocation, but has no effect on the size and does not alter its elements.
            After this function is called, the size becomes equal to the capacity.
            If n is zero, it releases all of the memory allocated in this instance.
            @param[in] n The new size
        */
        void resize( const size_t n );

        /// @brief The function to shrink the instance
        /**
            It reduces its capacity to fit its size.
            It may cause a re-allocation, but has no effect on the size and does not alter its elements.
            After this function is called, the size becomes equal to the capacity.
        */
        void shrink();

        /// @brief The function to copy all of the elements
        /**
            It copies all of the elements of the other array instance into this one.
			@return true if success, and false otherwise
        */
        bool copyFrom( const Array<TT>& a );

        /// @brief The function to copy all of the elements
        /**
            It copies all of the elements of this array instance into other one.
			@return true if success, and false otherwise
        */
        bool copyTo( Array<TT>& a ) const;

        /// @brief The assignment operator
        /**
            It creates a new reference of the given array.
            @param[in] a The array object to copy from
            @return The reference of the this array object
        */
        Array<TT>& operator=( const Array<TT>& a );

        /// @brief The equality operator
        /**
            It returns true if this array instance has the same elements and order as the given array instance.
            @param[in] other The array to be compared
            @return true if same, and false otherwise
        */
        bool operator==( const Array<TT>& a );

        /// @brief The inequality operator
        /**
            It returns true if this array instance does not have the same elements and order as the given array instance.
            @param[in] other The array to be compared
            @return true if different, and false otherwise
        */
        bool operator!=( const Array<TT>& a );

        /// @brief The reference of the first element.
        /**
            It returns the reference of the first element.
            @return The reference of the first element.
        */
        TT& first();

        /// @brief The value of the first element.
            /**
            It returns the value of the first element.
            @return The value of the first element.
        */
        const TT& first() const;

        /// @brief The reference of the last element.
        /**
            It returns the reference of the last element.
            @return The reference of the last element.
        */
        TT& last();

        /// @brief The value of the last element.
            /**
            It returns the value of the last element.
            @return The value of the last element.
        */
        const TT& last() const;

        /// @brief The function to add a new element
        /**
            It adds a new element to the end of the array.
            @param[in] v The value for the new last element being added
            @return The reference of the newly added last element
        */
        void append( const TT& v );

        /// @brief The function to add new elements
        /**
            It adds the elements of the other array instance to the end of this array instance.
            @param[in] elements The array being added at the end.
        */
        void append( const Array<TT>& a );

        /// @brief The function to insert a new element
        /**
            @param[in] index The index where the new element is inserted.
            @param[in] v A value to be copied to the inserted elements
        */
        void insert( const size_t& index, const TT& v );

        /// @brief The function to erase an element with the given index.
        /**
            @pram[in] index The index to be removed from the array.
        */
        void erase( const size_t& index );

        /// @brief The function to make every elements zero
        /**
            It set the block of the memory to zero so that the every element of the array instance has zero bits.
        */
        void zeroize();

        /// @brief The function to set all the elements as the given value
        /**
            It assigns the given value to all the elements of this array instance.
            @param[in] v The value for all the elements
        */
        void setValueAll( const TT& v );

        /// @brief The function to set all the elements as offset values.
        /**
            It assigns the increasing values to the elements of this array instance.
            @param[in] start The start value
            @param[in] offset The offset value
        */
        void setOffsetValues( const TT& start, const TT& offset );

        /// @brief The function to set all the elements as random values.
        /**
            It assigns the random values to all the elements of this array instance.
            @param[in] min The minimum value of the random range
            @param[in] max The maximum value of the random range
        */
        void setRandomValues( const TT& min, const TT& max, const size_t seed=0 );

        /// @brief The function to inverse
        /**
            It inverses the elements of this array instance.
        */
        void inverse();

        /// @brief The function to revers
        /**
            It reverses the order of the elements of this array instance.
        */
        void reverse();

        /// @brief The function to sort the elements.
        /**
            It sorts the elements of this array instance.
            @param[in] increasingOrder If true, it sorts in increasing order. If false, it sorts in decreasing order.
        */
        void sort( bool increasingOrder=true );

        /// @brief The function to shuffle the elements.
        /**
            It reorders the elements of this array instance.
        */
        void shuffle( size_t seed=0 );

        /// @brief The function to return the minimum value.
        /**
            It returns the minimum value of this array instance.
            @return The minimum value
        */
        TT minValue() const;

        /// @brief The function to return the maximum value.
        /**
            It returns the maximum value of this array instance.
            @return The maximum value
        */
        TT maxValue() const;

        // Caution)
        // The first remove() function is for removing elements by sharing deleteMask array.
        // "numToDelete" must be set as the returning value of the buildDeleteMask() function.
        size_t buildDeleteMask( const Array<size_t>& indicesToBeDeleted, Array<char>& deleteMask ) const;
        size_t eliminate( const Array<char>& deleteMask, size_t numToDelete );
        size_t eliminate( const Array<size_t>& indicesToBeDeleted );
        size_t eliminate( const size_t n );

        size_t remove( const Array<char>& deleteMask );

        size_t deduplicate();

		void write( std::ofstream& fout ) const;
		void read ( std::ifstream& fin  );

        /// @brief The function to save data to file
		/**
			It saves the data of the array into a file.
			@param[in] filePathName The file path and name to save the data into it.
			@return True if success, and false otherwise.
		*/
		bool save( const char* filePathName ) const;

        /// @brief The function to load data from file
		/**
			It loads the data of the array from a file.
			@param[in] filePathName The file path and name to load the data from it.
			@return True if success, and false otherwise.
		*/
        bool load( const char* filePathName );

        void printFile( const char* name );

        void print( const bool horizontally=false, const bool withIndex=true, const size_t maxIndex=ULONG_MAX ) const;

        /////////////////////////////
        // inline member functions //

        BORA_UNIFIED const size_t num() const { return _size; }
        BORA_UNIFIED const size_t size() const { return _size; }
        BORA_UNIFIED const size_t length() const { return _size; }        

        BORA_UNIFIED const size_t capacity() const { return _capacity; }

        BORA_UNIFIED const MemorySpace memorySpace() const { return _memorySpace; }

        BORA_UNIFIED const float increasingRatio() const { return _increasingRatio; }

        BORA_UNIFIED TT* pointer() const { return _data; }
        BORA_UNIFIED TT** pPointer() { return &_data; }

        BORA_UNIFIED TT* begin() { return _data; }
        BORA_UNIFIED const TT* begin() const { return _data; }

        BORA_UNIFIED TT* end() { return ( _data + _size ); }
        BORA_UNIFIED const TT* end() const { return ( _data + _size ); }

        BORA_UNIFIED TT& at( const size_t& i ) { return _data[i]; }
        BORA_UNIFIED const TT& at( const size_t& i ) const { return _data[i]; }

        BORA_UNIFIED TT& operator[]( const size_t& i ) { return _data[i]; }
        BORA_UNIFIED const TT& operator[]( const size_t& i ) const { return _data[i]; }

        BORA_UNIFIED TT& pop_back() { return _data[--_size]; }

        /// @brief The function to exchange the given arrays.
        /**
            It swaps the given arrays.
            @param[in] a The first array
            @param[in] b The second array
        */
        static void exchange( Array<TT>& a, Array<TT>& b )
        {
            Swap( a._memorySpace,     b._memorySpace     );
            Swap( a._data,            b._data            );
            Swap( a._size,            b._size            );
            Swap( a._capacity,        b._capacity        );
            Swap( a._increasingRatio, b._increasingRatio );          
            Swap( a._referenceMode,   b._referenceMode   );
        }

        void swap( Array<TT>& a )
        {
            Array::exchange( *this, a );
        }

        size_t typeSize() const
        {
            return sizeof(TT);
        }

        std::string dataType() const
        {
            std::string type( "Array_" );
            return ( type + typeid(TT).name() );
        }

    private:

        TT* allocate( const size_t n );
        void release( TT* ptr );
        void memcopy( TT* dst, TT* src, const size_t n );
        void memzero( TT* dst, const size_t n );
};

////////////////////////////////////
// member function implementation //

template <class TT>
BORA_UNIFIED
Array<TT>::Array()
{
    // nothing to do
}

template <class TT>
BORA_UNIFIED
Array<TT>::Array( const Array<TT>& a )
{
    // They will cause problems for CUDA.
    // Array::operator=( a );
    // *this = a;

    _memorySpace     = a._memorySpace;
    _data            = a._data;
    _size            = a._size;
    _capacity        = a._capacity;
    _increasingRatio = a._increasingRatio;
    _referenceMode   = true;
}

template <class TT>
Array<TT>::Array( const size_t n, const MemorySpace memorySpace )
{
    Array::initialize( n, memorySpace );
}

template <class TT>
BORA_UNIFIED
Array<TT>::~Array()
{
    if( _referenceMode == false )
    {
        Array::finalize();
    }
}

template <class TT>
void Array<TT>::initialize( const size_t n, const MemorySpace memorySpace )
{
    if( ( _size == n ) && ( _memorySpace == memorySpace ) ) { return; }

    Array::finalize();

    _memorySpace = memorySpace;

    if( n > 0 )
    {
        _data = Array::allocate( _size = _capacity = n );

        Array::zeroize();
    }
}

template <class TT>
void Array<TT>::finalize()
{
    Array::release( _data );

    _memorySpace     = kHost;
    _data            = (TT*)NULL;
    _size            = 0;
    _capacity        = 0;
    _increasingRatio = 2.f;
    _referenceMode   = false;
}

template <class TT>
void Array<TT>::clear()
{
    _size = 0;
}

template <class TT>
void Array<TT>::reserve( const size_t n )
{
    if( _capacity >= n ) { return; }

    TT* tmp = Array::allocate( n );

    Array::memcopy( tmp, _data, _size );

    Array::release( _data );

    _data = tmp;

    _capacity = n;
}

template <class TT>
void Array<TT>::resize( const size_t n )
{
    if( n == 0 )
    {
        Array::release( _data );

        _data     = (TT*)NULL;
        _size     = 0;
        _capacity = 0;

        return;
    }

    TT* tmp = Array::allocate( n );

    const size_t m = Min( n, _size );

    Array::memcopy( tmp, _data, m );

    Array::release( _data );

    _data = tmp;

    _size = _capacity = n;
}

template <class TT>
void Array<TT>::shrink()
{
    if( ( _size == _capacity ) || ( _size == 0 ) ) { return; }

    TT* tmp = Array::allocate( _size );

    Array::memcopy( tmp, _data, _size );

    Array::release( _data );

    _data = tmp;

    _capacity = _size;
}

template <class TT>
bool Array<TT>::copyFrom( const Array<TT>& a )
{
    const size_t n = a._size;

    if( n == 0 ) { Array::clear(); return true; }

    const MemorySpace srcKind = a._memorySpace;
    const MemorySpace dstKind = _memorySpace;

    Array::initialize( n, dstKind );

    cudaMemcpyKind kind = cudaMemcpyHostToHost;

    if     ( ( srcKind == kHost    ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyHostToHost;     }
    else if( ( srcKind == kHost    ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyHostToDevice;   }
    else if( ( srcKind == kHost    ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kDevice  ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyDeviceToHost;   }
    else if( ( srcKind == kDevice  ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyDeviceToDevice; }
    else if( ( srcKind == kDevice  ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else { return false; }

    cudaMemcpy( (void*)_data, (const void*)a._data, sizeof(TT)*n, kind );

    return true;
}

template <class TT>
bool Array<TT>::copyTo( Array<TT>& a ) const
{
    const size_t n = _size;

    if( n == 0 ) { a.clear(); return true; }

    const MemorySpace srcKind = _memorySpace;
    const MemorySpace dstKind = a._memorySpace;

    a.initialize( n, dstKind );

    cudaMemcpyKind kind = cudaMemcpyHostToHost;

    if     ( ( srcKind == kHost    ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyHostToHost;     }
    else if( ( srcKind == kHost    ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyHostToDevice;   }
    else if( ( srcKind == kHost    ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kDevice  ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyDeviceToHost;   }
    else if( ( srcKind == kDevice  ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyDeviceToDevice; }
    else if( ( srcKind == kDevice  ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kHost    ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kDevice  ) ) { kind = cudaMemcpyDefault;        }
    else if( ( srcKind == kUnified ) && ( dstKind == kUnified ) ) { kind = cudaMemcpyDefault;        }
    else { return false; }

    cudaMemcpy( (void*)a._data, (const void*)_data, sizeof(TT)*n, kind );

    return true;
}

template <class TT>
Array<TT>& Array<TT>::operator=( const Array<TT>& a )
{
    _memorySpace     = a._memorySpace;
    _data            = a._data;
    _size            = a._size;
    _capacity        = a._capacity;
    _increasingRatio = a._increasingRatio;
    _referenceMode   = true;

    return (*this);
}

template <class TT>
bool Array<TT>::operator==( const Array<TT>& a )
{
    if( _memorySpace != a._memorySpace )
    {
        COUT << "Error@Array::operator==(): Memory space mismatch." << ENDL;
        return false;
    }

    if( _size != a._size ) { return false; }

    for( size_t i=0; i<_size; ++i )
    {
        if( _data[i] != a._data[i] )
        {
            return false;
        }
    }

    return true;
}

template <class TT>
bool Array<TT>::operator!=( const Array<TT>& a )
{
    if( _memorySpace != a._memorySpace )
    {
        COUT << "Error@Array::operator!=(): Memory space mismatch." << ENDL;
        return true;
    }

    if( _size != a._size ) { return true; }

    for( size_t i=0; i<_size; ++i )
    {
        if( _data[i] != a._data[i] )
        {
            return true;
        }
    }

    return false;
}

template <class TT>
TT& Array<TT>::first()
{
    return _data[0];
}

template <class TT>
const TT& Array<TT>::first() const
{
    return _data[0];
}

template <class TT>
TT& Array<TT>::last()
{
    return _data[_size-1];
}

template <class TT>
const TT& Array<TT>::last() const
{
    return _data[_size-1];
}

template <class TT>
void Array<TT>::append( const TT& v )
{
    if( _size == _capacity )
    {
        Array::reserve( Max( size_t(_size*_increasingRatio), size_t(1) ) );
    }

    _data[ _size++ ] = v;
}

template <class TT>
void Array<TT>::append( const Array<TT>& a )
{
    Array::reserve( _size + a.size() );

    Array::memcopy( &_data[_size], a._data, a._size );

    _size += a._size;
}

template <class TT>
void Array<TT>::insert( const size_t& index, const TT& v )
{
    if( index > _size )
    {
        COUT << "Error@Array::insert(): Invalid index." << ENDL;
        return;
    }

    const size_t finalSize = _size + 1;

    if( finalSize > _capacity )
    {
        Array::reserve( Max( size_t(finalSize*_increasingRatio), size_t(1) ) );
    }

    for( size_t i=_size; i>index; --i )
    {
        _data[i] = _data[i-1];
    }

    _data[index] = v;

    _size = finalSize;
}

template <class TT>
void Array<TT>::erase( const size_t& index )
{
    if( index >= _size )
    {
        COUT << "Error@Array::erase(): Invalid index." << ENDL;
        return;
    }

    const size_t finalSize = _size - 1;

    for( size_t i=index; i<finalSize; ++i )
    {
        _data[i] = _data[i+1];
    }

    _size = finalSize;
}

template <class TT>
void Array<TT>::zeroize()
{
    Array::memzero( _data, _size );
}

template <class TT>
void Array<TT>::setValueAll( const TT& v )
{
    if( _data )
    {
        for( int n=0; n<_size; ++n )
        {
            _data[n] = v;
        }
    }
}

template <class TT>
void Array<TT>::setOffsetValues( const TT& start, const TT& offset )
{
    for( size_t i=0; i<_size; ++i )
    {
        _data[i] = start + ( (TT)i * offset );
    }
}

template <class TT>
void Array<TT>::setRandomValues( const TT& min, const TT& max, const size_t seed )
{
    const TT diff = max - min;
    
    for( size_t i=0; i<_size; ++i )
    {
        _data[i] = diff * (TT)Rand( seed+i ) + min;
    }
}

template <class T>
void Array<T>::inverse()
{
    for( size_t i=0; i<_size; ++i )
    {
        T& v = _data[i];
        v = !v;
    }
}

template <class T>
void Array<T>::reverse()
{
    const size_t n = (size_t)(_size/2);

    for( size_t i=0; i<n; ++i )
    {
        Swap( _data[i], _data[_size-1-i] );
    }
}

template <class T>
void Array<T>::sort( bool increasingOrder )
{
    for( size_t i=0; i<_size; ++i )
    {
        T& vi = _data[i];

        for( size_t j=i+1; j<_size; ++j )
        {
            T& vj = _data[j];

            if( increasingOrder )
            {
                if( vi > vj ) { Swap( vi, vj ); }
            }
            else // decreasing order
            {
                if( vi < vj ) { Swap( vi, vj ); }
            }
        }
    }
}

template <class TT>
void Array<TT>::shuffle( size_t seed )
{
    for( size_t i=1; i<_size; ++i )
    {
        const size_t j = RandInt( seed++, 0, i );

        Swap( _data[i], _data[j] );
    }
}

template <class TT>
TT Array<TT>::minValue() const
{
    bool validType = false;
    TT ret = (TT)0;

    if( !validType ) if( typeid(TT).name() == typeid(bool          ).name() ) { validType = true; ret = 1;         }
    if( !validType ) if( typeid(TT).name() == typeid(char          ).name() ) { validType = true; ret = CHAR_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned char ).name() ) { validType = true; ret = UCHAR_MAX; }
    if( !validType ) if( typeid(TT).name() == typeid(short         ).name() ) { validType = true; ret = SHRT_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned short).name() ) { validType = true; ret = USHRT_MAX; }
    if( !validType ) if( typeid(TT).name() == typeid(int           ).name() ) { validType = true; ret = INT_MAX;   }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned int  ).name() ) { validType = true; ret = UINT_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(long          ).name() ) { validType = true; ret = LONG_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned long ).name() ) { validType = true; ret = ULONG_MAX; }
    if( !validType ) if( typeid(TT).name() == typeid(float         ).name() ) { validType = true; ret = FLT_MAX;   }
    if( !validType ) if( typeid(TT).name() == typeid(double        ).name() ) { validType = true; ret = DBL_MAX;   }
    if( !validType ) if( typeid(TT).name() == typeid(long double   ).name() ) { validType = true; ret = LDBL_MAX;  }

    if( !validType )
    {
        COUT << "Error@Array:min(): Invalid data type." << ENDL;
        return 0;
    }

    for( size_t i=0; i<_size; ++i )
    {
        const TT& v = _data[i];

        if( v < ret ) { ret = v; }
    }

    return ret;
}

template <class TT>
TT Array<TT>::maxValue() const
{
    bool validType = false;
    TT ret = (TT)0;

    if( !validType ) if( typeid(TT).name() == typeid(bool          ).name() ) { validType = true; ret = (TT)0;      }
    if( !validType ) if( typeid(TT).name() == typeid(char          ).name() ) { validType = true; ret = CHAR_MIN;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned char ).name() ) { validType = true; ret = (TT)0;      }
    if( !validType ) if( typeid(TT).name() == typeid(short         ).name() ) { validType = true; ret = SHRT_MIN;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned short).name() ) { validType = true; ret = (TT)0;      }
    if( !validType ) if( typeid(TT).name() == typeid(int           ).name() ) { validType = true; ret = INT_MIN;   }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned int  ).name() ) { validType = true; ret = (TT)0;      }
    if( !validType ) if( typeid(TT).name() == typeid(long          ).name() ) { validType = true; ret = LONG_MIN;  }
    if( !validType ) if( typeid(TT).name() == typeid(unsigned long ).name() ) { validType = true; ret = (TT)0;      }
    if( !validType ) if( typeid(TT).name() == typeid(float         ).name() ) { validType = true; ret = -FLT_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(double        ).name() ) { validType = true; ret = -DBL_MAX;  }
    if( !validType ) if( typeid(TT).name() == typeid(long double   ).name() ) { validType = true; ret = -LDBL_MAX; }

    if( !validType )
    {
        COUT << "Error@Array:max(): Invalid data type." << ENDL;
        return 0;
    }

    for( size_t i=0; i<_size; ++i )
    {
        const TT& v = _data[i];

        if( v > ret ) { ret = v; }
    }

    return ret;
}

template <class TT>
size_t Array<TT>::buildDeleteMask( const Array<size_t>& indicesToBeDeleted, Array<char>& deleteMask ) const
{
    const size_t n = indicesToBeDeleted.size();

    if( ( _size == 0 ) || ( n == 0 ) )
    {
        deleteMask.clear();
        return 0;
    }

    size_t numToDelete = 0;

    deleteMask.resize( _size );
    deleteMask.zeroize();

    for( size_t i=0; i<n; ++i )
    {
        const size_t& idx = indicesToBeDeleted[i];

        if( idx >= _size ) { continue; } // out of index

        if( deleteMask[idx] ) { continue; } // already checked

        deleteMask[idx] = true;

        ++numToDelete;
    }

    return numToDelete;
}

template <class T>
size_t Array<T>::eliminate( const Array<char>& deleteMask, size_t numToDelete )
{
    if( ( _size == 0 ) || ( deleteMask.size() == 0 ) )
    {
        return _size;
    }

    if( _size != (size_t)deleteMask.size() )
    {
        COUT << "Error@Array::eliminate(): Invalid array size." << ENDL;
        return _size;
    }

    if( _size == numToDelete )
    {
        Array::clear();
        return 0;
    }

    const size_t finalSize = _size - numToDelete;

    Array<T> tmp( finalSize, _memorySpace );

    for( size_t i=0, count=0; i<_size; ++i )
    {
        if( deleteMask[i] ) { continue; }

        tmp[count++] = _data[i];
    }

    Array::exchange( *this, tmp );

    return finalSize;
}

template <class TT>
size_t Array<TT>::eliminate( const Array<size_t>& indicesToBeDeleted )
{
    Array<char> deleteMask;
    const size_t numToDelete = Array::buildDeleteMask( indicesToBeDeleted, deleteMask );

    return Array::eliminate( deleteMask, numToDelete );
}

template <class TT>
size_t Array<TT>::eliminate( const size_t n )
{
    if( n >= _size || _size == 0 ) return _size-1;
    _data[n] = _data[--_size];
}

template <class TT>
size_t Array<TT>::remove( const Array<char>& deleteMask )
{
    size_t end = _size-1;
    size_t delCount = 0;

    for( size_t n=0; n<deleteMask.size(); ++n )
    {
        if( n >= _size ) break;

        if( deleteMask[n] )
        {
            delCount++;

            for( size_t m=end; m>=n+1; --m )
            {
                if( deleteMask[m] == 0 )
                {
                    _data[n] = _data[m];
                    end = m-1;
                    break;
                }
            }
        }
    }

    _size -= delCount;

    return _size;
}

template <class TT>
size_t Array<TT>::deduplicate()
{
    size_t numToDelete = 0;

    Array<char> deleteMask( _size );
    deleteMask.zeroize();

    for( size_t i=0; i<_size; ++i )
    {
        if( deleteMask[i] ) { continue; }

        const TT& vi = _data[i];

        for( size_t j=i+1; j<_size; ++j )
        {
            const TT& vj = _data[j];

            if( vi == vj )
            {
                deleteMask[j] = true;
                ++numToDelete;
            }
        }
    }

    return Array::eliminate( deleteMask, numToDelete );
}

template <class TT>
void Array<TT>::write( std::ofstream& fout ) const
{
    if( _memorySpace == kDevice )
    {
        COUT << "Error@Array::write(): Not supported for the device array." << ENDL;
        return;
    }

    const size_t& n = _size;
    fout.write( (char*)&n, sizeof(size_t) );

    if( n )
    {
        fout.write( (char*)_data, sizeof(TT)*n );
    }
}

template <class TT>
void Array<TT>::read ( std::ifstream& fin  )
{
    if( _memorySpace == kDevice )
    {
        COUT << "Error@Array::read(): Not supported for the device array." << ENDL;
        return;
    }

    size_t n = 0;
    fin.read( (char*)&n, sizeof(size_t) );

    if( n )
    {
        Array::resize( n );
        fin.read( (char*)_data, sizeof(TT)*n );
    }
    else
    {
        Array::finalize();
    }
}

template <class TT>
bool Array<TT>::save( const char* filePathName ) const
{
    if( _memorySpace == kDevice )
    {
        COUT << "Error@Array::save(): Not supported for the device array." << ENDL;
        return false;
    }

    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@Array::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    Array::write( fout );

    fout.close();

    return true;
}

template <class TT>
bool Array<TT>::load( const char* filePathName )
{
    Array::clear();
    
    if( _memorySpace == kDevice )
    {
        COUT << "Error@Array::load(): Not supported for the device array." << ENDL;
        return false;
    }

    std::ifstream fin( filePathName, std::ios::in|std::ios::binary );

    if( fin.fail() )
    {
        COUT << "Error@Array::load(): Failed to load file." << ENDL;
        return false;
    }

    Array::read( fin );

    fin.close();

    return true;
}

template <class TT>
void Array<TT>::printFile( const char* name )
{
    std::ofstream file;
    file.open( name );

    for( size_t i=0; i<_size; ++i )
    {
        file << _data[i];
        file << '\n';
    }

    file.close();
}

template <class TT>
void Array<TT>::print( const bool horizontally, const bool withIndex, const size_t maxIndex ) const
{
    if( _memorySpace != kHost )
    {
        COUT << "Error@Array::print(): Not supported for the non-host array." << ENDL;
        return;
    }

    const size_t n = Min( _size, maxIndex );

    if( horizontally )
    {
        for( size_t i=0; i<n; ++i )
        {
            if( withIndex ) { COUT << i << ": " << _data[i] << " "; }
            else { COUT << _data[i] << " "; }
        }
        COUT << ENDL;
    }
    else // vertically
    {
        for( size_t i=0; i<n; ++i )
        {
            if( withIndex ) { COUT << i << ": " << _data[i] << ENDL; }
            else { COUT << _data[i] << ENDL; }
        }
    }

    COUT << ENDL;
}

template <class TT>
inline TT* Array<TT>::allocate( const size_t n )
{
    if( n == 0 ) { return 0; }

    TT* ptr( NULL );

    if     ( _memorySpace == kHost    ) { ptr = new TT[n];                         }
    else if( _memorySpace == kDevice  ) { cudaMalloc       ( &ptr, sizeof(TT)*n ); }
    else if( _memorySpace == kUnified ) { cudaMallocManaged( &ptr, sizeof(TT)*n ); }

    return ptr;
}

template <class TT>
inline void Array<TT>::release( TT* ptr )
{
    if( !ptr ) { return; }

    if     ( _memorySpace == kHost    ) { delete[] ptr;    }
    else if( _memorySpace == kDevice  ) { cudaFree( ptr ); }
    else if( _memorySpace == kUnified ) { cudaFree( ptr ); }
}

template <class TT>
inline void Array<TT>::memcopy( TT* dst, TT* src, const size_t n )
{
    if( n == 0 ) { return; }

    if     ( _memorySpace == kHost    ) { memcpy    ( dst, src, sizeof(TT)*n );                           }
    else if( _memorySpace == kDevice  ) { cudaMemcpy( dst, src, sizeof(TT)*n, cudaMemcpyDeviceToDevice ); }
    else if( _memorySpace == kUnified ) { cudaMemcpy( dst, src, sizeof(TT)*n, cudaMemcpyDefault );        }
}

template <class TT>
inline void Array<TT>::memzero( TT* dst, const size_t n )
{
    if( n == 0 ) { return; }

    if     ( _memorySpace == kHost    ) { memset    ( dst, 0, sizeof(TT)*n ); }
    else if( _memorySpace == kDevice  ) { cudaMemset( dst, 0, sizeof(TT)*n ); }
    else if( _memorySpace == kUnified ) { cudaMemset( dst, 0, sizeof(TT)*n ); }
}

template <class TT>
std::ostream& operator<<( std::ostream& os, const Array<TT>& object )
{
	os << "<Array>" << ENDL;
	os << " Data Type       : " << object.dataType()        << ENDL;
	os << " Size            : " << object.size()            << ENDL;
	os << " Capacity        : " << object.capacity()        << ENDL;
    os << " Memory Space    : " << object.memorySpace()     << ENDL;
    os << " Increasing Ratio: " << object.increasingRatio() << ENDL;
	os << ENDL;
	return os;
}

typedef Array<char>           CharArray;
typedef Array<unsigned char>  UCharArray;
typedef Array<int16_t>        Int16Array;
typedef Array<uint16_t>       UInt16Array;
typedef Array<int32_t>        Int32Array;
typedef Array<uint32_t>       UInt32Array;
typedef Array<int64_t>        Int64Array;
typedef Array<uint64_t>       UInt64Array;
typedef Array<size_t>         IndexArray;
typedef Array<short>          ShortArray;
typedef Array<unsigned short> UShortArray;
typedef Array<int>            IntArray;
typedef Array<unsigned int>   UIntArray;
typedef Array<float>          FloatArray;
typedef Array<double>         DoubleArray;
typedef Array<Idx2>           Idx2Array;
typedef Array<Vec2i>          Vec2iArray;
typedef Array<Vec2f>          Vec2fArray;
typedef Array<Vec2d>          Vec2dArray;
typedef Array<Idx3>           Idx3Array;
typedef Array<Vec3i>          Vec3iArray;
typedef Array<Vec3f>          Vec3fArray;
typedef Array<Vec3f>          PointArray;
typedef Array<Vec3f>          VectorArray;
typedef Array<Vec3d>          Vec3dArray;
typedef Array<Idx4>           Idx4Array;
typedef Array<Vec4i>          Vec4iArray;
typedef Array<Vec4f>          Vec4fArray;
typedef Array<Vec4d>          Vec4dArray;
typedef Array<Quatf>          QuatfArray;
typedef Array<Quatd>          QuatdArray;
typedef Array<Mat22f>         Mat22fArray;
typedef Array<Mat22d>         Mat22dArray;
typedef Array<Mat33f>         Mat33fArray;
typedef Array<Mat33d>         Mat33dArray;
typedef Array<Mat44f>         Mat44fArray;
typedef Array<Mat44d>         Mat44dArray;
typedef Array<AABB2f>         AABB2fArray;
typedef Array<AABB2d>         AABB2dArray;
typedef Array<AABB3f>         AABB3fArray;
typedef Array<AABB3d>         AABB3dArray;

BORA_NAMESPACE_END

#endif

