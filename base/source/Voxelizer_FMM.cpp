//----------------//
// Voxelizer_FMM.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2013.10.24                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Voxelizer_FMM::Voxelizer_FMM()
{
	reset();
}

void
Voxelizer_FMM::reset()
{

	//Grid::reset();
	_grid = Grid(); // TODO: Grid::reset() ?

	_initialState = true;
	_onCell = true;

	_iMax = _jMax = _kMax = 0;
	_eps  = 0.001f;
	_negRange = _posRange = 0.f;

	_posHeap.clear();
	_negHeap.clear();

	_lvs  = (ScalarDenseField*)NULL;
	//_vel  = (ZVectorField3D*)NULL;
	_stt  = (MarkerDenseField*)NULL;

	_vPos = (Vec3fArray*)NULL;
	//_vVel = (ZVectorArray*)NULL;
	_v012 = (IndexArray*)NULL;

}

void
Voxelizer_FMM::set( const Grid& grid )
{
	Voxelizer_FMM::reset();
	_grid = grid;
	AABB3f aabb = _grid.boundingBox();
	float Lx = aabb.width(0);
	float Ly = aabb.width(1);
	float Lz = aabb.width(2);
	float dx = Lx/float(_grid.nx());
	float dy = Ly/float(_grid.ny());
	float dz = Lz/float(_grid.nz());
	_h = (dx+dy+dz)/3.f;
	//Grid::operator=( grid );
	//_h = Grid::avgCellSize(); //change bc this function doesnt exist
}

void
Voxelizer_FMM::set( float h, int maxSubdivision, const AABB3f& bBox )
{
	Voxelizer_FMM::reset();
	//Grid::set( h, maxSubdivision, bBox );
	_grid = Grid( maxSubdivision, bBox );
	_h = h;
}

void
Voxelizer_FMM::addMesh( ScalarDenseField& lvs, MarkerDenseField& stt, TriangleMesh& mesh, float negRange, float posRange )
{
//	if( !lvs.location()==ZFieldLocation::zNode || !lvs.location()==ZFieldLocation::zCell )
//	{
//		cout << "Error@ZVoxelizer::addMesh(): Invalid field location." << endl;
//		return;
//	}
//
//	if( !lvs.directComputable(stt) )
//	{
//		cout << "Error@ZVoxelizer::addMesh(): Not computable fields." << endl;
//		return;
//	}

	if( mesh.numVertices() < 3 )
	{
		lvs.setValueAll( BORA_LARGE );
		return;
	}

	//_onCell = (lvs.location()==ZFieldLocation::zCell) ? true : false;
	_onCell = true; 

	_iMax = lvs.nx()-1;
	_jMax = lvs.ny()-1;
	_kMax = lvs.nz()-1;

	_eps  = 0.001f; // epsilon in voxel space
	_negRange = Abs( negRange );
	_posRange = Abs( posRange );

	_lvs  = &lvs;
	//_vel  = (ZVectorField3D*)NULL;
	_stt  = &stt;

	_vPos = &mesh.pos;
	//_vVel = (ZVectorArray*)NULL;
	_v012 = &mesh.indices;

	_tagInterfacialElements();
}

void
Voxelizer_FMM::_tagInterfacialElements()
{
	ScalarDenseField& 	lvs = *_lvs;
	MarkerDenseField& 	stt = *_stt;
	Vec3fArray&   		vPos = *_vPos;
	IndexArray&   	 	v012 = *_v012;

//	const bool hasVel = ( _vel && _vVel );
//	ZVector* vel  = hasVel ? (ZVector*)(_vel->pointer())  : (ZVector*)NULL;
//	ZVector* vVel = hasVel ? (ZVector*)(_vVel->pointer()) : (ZVector*)NULL;

	const int numVertices = vPos.capacity();
	const int numTrifaces = v012.length()/3;
	if( !numVertices || !numTrifaces ) { return; }

	if( _initialState )
	{
		if( _lvs ) { lvs.setValueAll( _iMax+_jMax+_kMax ); }
		//if( _vel ) { _vel->zerorize();              }
		if( _stt ) { stt.setValueAll( FMMState::far );   } 

		_initialState = false;
	}

	int     idx, sgn;
	int     i,i0,i1, j,j0,j1, k,k0,k1;
	float   lvsEst;
	double  x,y,z;
	Vec3f	baryCoords;
	Vec3f  	minP, maxP;
	Vec3f  	p, p0,p1,p2;
	Vec3f 	triNrm, velEst, v0,v1,v2;

	AABB3f aabb = lvs.boundingBox();
	float Lx = aabb.width(0);
	float Ly = aabb.width(1);
	float Lz = aabb.width(2);
	float dx = Lx/float(lvs.nx());
	float dy = Ly/float(lvs.ny());
	float dz = Lz/float(lvs.nz());

	const Vec3f minPt = aabb.minPoint(); // of boundingbox? get that first?
	const float ddx =1/dx,     ddy =1/dy,     ddz =1/dz;
	const float dx_2=0.5f*dx,  dy_2=0.5f*dy,  dz_2=0.5f*dz;

	for( int iTri=0; iTri<numTrifaces; iTri++ )
	{
		const int& v0 = v012[iTri*3  ];
		const int& v1 = v012[iTri*3+1];
		const int& v2 = v012[iTri*3+2];
		p0=vPos[v0]; p1=vPos[v1]; p2=vPos[v2];
		//if( hasVel ) { v0=vVel[v[0]]; v1=vVel[v[1]]; v2=vVel[v[2]]; }

		triNrm = _normal( p0,p1,p2 );

		GetMinMax( p0.x,p1.x,p2.x, minP.x,maxP.x );
		GetMinMax( p0.y,p1.y,p2.y, minP.y,maxP.y );
		GetMinMax( p0.z,p1.z,p2.z, minP.z,maxP.z );

		Vec3f minVox, maxVox;		
		minVox = lvs.voxelPoint( minP );
		maxVox = lvs.voxelPoint( maxP );

		Idx3 minCellInds, maxCellInds;
		lvs.getCellIndices( minVox, &minCellInds.x, &minCellInds.y, &minCellInds.z ); 
		lvs.getCellIndices( maxVox, &maxCellInds.x, &maxCellInds.y, &maxCellInds.z );
		i0 = minCellInds.x; i1 = maxCellInds.x;
		j0 = minCellInds.y; j1 = maxCellInds.y;
		k0 = minCellInds.z; k1 = maxCellInds.z;

		i0 = Clamp( i0, 0, _iMax );
		i1 = Clamp( i1, 0, _iMax );
		j0 = Clamp( j0, 0, _jMax );
		j1 = Clamp( j1, 0, _jMax );
		k0 = Clamp( k0, 0, _kMax );
		k1 = Clamp( k1, 0, _kMax );

		if( Abs(triNrm.x) > _eps )
		{
			sgn = (triNrm.x>0) ? (+1) : (-1);

			for( k=k0; k<=k1; ++k )
			for( j=j0; j<=j1; ++j )
			{{
				p = lvs.worldPoint(Vec3f(0,j,k));

				if( _baryCoords( p, p0,p1,p2, 1, baryCoords ) )
				{
					x = ( baryCoords[0]*p0.x + baryCoords[1]*p1.x + baryCoords[2]*p2.x ) - minPt.x;

					//if( hasVel ) { velEst = WeightedSum( v0,v1,v2, baryCoords ); }

					const int I = i = int(x*ddx);
					if( i >= 0 && i <= _iMax )
					{
						if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
						{
							lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (i=I+1) <= _iMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (i=I+2) <= _iMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (i=(I-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}
						}
					}
				}
			}}
		}

		if( Abs(triNrm.y) > _eps )
		{
			sgn = (triNrm.y>0) ? (+1) : (-1);

			for( k=k0; k<=k1; ++k )
			for( i=i0; i<=i1; ++i )
			{{
				p = lvs.worldPoint(Vec3f(i,0,k));

				if( _baryCoords( p, p0,p1,p2, 2, baryCoords ) )
				{
					y = ( baryCoords[0]*p0.y + baryCoords[1]*p1.y + baryCoords[2]*p2.y ) - minPt.y;

					//if( hasVel ) { velEst = WeightedSum( v0,v1,v2, baryCoords ); }

					const int J = j = int(y*ddy);
					if( j >= 0 && j <= _jMax )
					{
						if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
						{
							lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (j=J+1) <= _jMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (j=J+2) <= _jMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (j=(J-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}
						}
					}
				}
			}}
		}

		if( Abs(triNrm.z) > _eps )
		{
			sgn = (triNrm.z>0) ? (+1) : (-1);

			for( j=j0; j<=j1; ++j )
			for( i=i0; i<=i1; ++i )
			{{
				p = lvs.worldPoint(Vec3f(i,j,0));

				if( _baryCoords( p, p0,p1,p2, 0, baryCoords ) )
				{
					z = ( baryCoords[0]*p0.z + baryCoords[1]*p1.z + baryCoords[2]*p2.z ) - minPt.z;

					//if( hasVel ) { velEst = WeightedSum( v0,v1,v2, baryCoords ); }

					const int K = k = int(z*ddz);
					if( k >= 0 && k <= _kMax )
					{
						if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
						{
							lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (k=K+1) <= _kMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (k=K+2) <= _kMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (k=(K-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FMMState::interface; //if(hasVel){vel[idx]=velEst;}
							}
						}
					}
				}
			}}
		}
	}
}

void
Voxelizer_FMM::finalize()
{
	if( !_lvs || !_stt ) { return; }

	ScalarDenseField& lvs = *_lvs;
	MarkerDenseField& stt = *_stt;

	int idx0, idx1;

	// initialize two heaps (both the positive heap and the negative heap)
	for( int k=0; k<=_kMax; ++k ) { 
	for( int j=0; j<=_jMax; ++j ) { 
	for( int i=0; i<=_iMax; ++i ) {

		idx0 = lvs.cellIndex(i,j,k);
		if( stt[idx0] != FMMState::interface ) { continue; }

		if( i != 0 )
		{
			idx1 = lvs.i0(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i-1,j,k), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i-1,j,k), lvs[idx0]-1 ) ); }
			}
		}

		if( i != _iMax )
		{
			idx1 = lvs.i1(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i+1,j,k), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i+1,j,k), lvs[idx0]-1 ) ); }
			}
		}

		if( j != 0 )
		{
			idx1 = lvs.j0(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j-1,k), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i,j-1,k), lvs[idx0]-1 ) ); }
			}
		}

		if( j != _jMax )
		{
			idx1 = lvs.j1(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j+1,k), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i,j+1,k), lvs[idx0]-1 ) ); }
			}
		}

		if( k != 0 )
		{
			idx1 = lvs.k0(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j,k-1), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i,j,k-1), lvs[idx0]-1 ) ); }
			}
		}

		if( k != _kMax )
		{
			idx1 = lvs.k1(idx0);

			if( stt[idx1] == FMMState::far )
			{
				stt[idx1] = FMMState::trial;

				if( lvs[idx0] > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j,k+1), lvs[idx0]+1 ) ); }
				else                { _negHeap.push( HEAPNODE( Vec3i(i,j,k+1), lvs[idx0]-1 ) ); }
			}
		}
	}}}

	// update & marching
	Vec3i ijk;
	while( !_negHeap.empty() ) { ijk=_negHeap.top().data; _negHeap.pop(); _update(ijk,-1); }
	while( !_posHeap.empty() ) { ijk=_posHeap.top().data; _posHeap.pop(); _update(ijk,+1); }

	float minValue = lvs.minValue(); 
	float maxValue = lvs.maxValue();

//	// scale & set min/max.
//	const int nElems = lvs.numCells();
//	for( int i=0; i<nElems; i++ )
//	{
//		float& v = lvs[i];
//		v *= _h;
//
//		minValue = Min( minValue, v );
//		maxValue = Max( maxValue, v );
//	}

	reset();
}

void
Voxelizer_FMM::_update( const Vec3i& ijk, int sign )
{
	ScalarDenseField& lvs = *_lvs;
	//ZVectorField3D& vel = *_vel;
	MarkerDenseField& stt = *_stt;

	const int &i=ijk[0], &j=ijk[1], &k=ijk[2];
	const int idx0 = lvs.cellIndex( i, j, k );

	bool infoExist = false;
	float infoX=BORA_LARGE, infoY=BORA_LARGE, infoZ=BORA_LARGE;

	int idx1, cnt=0;
	Vec3f sum;
	if( i !=0     ) { idx1=lvs.i0(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoX=Min(infoX,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }
	if( i !=_iMax ) { idx1=lvs.i1(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoX=Min(infoX,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }
	if( j !=0     ) { idx1=lvs.j0(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoY=Min(infoY,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }
	if( j !=_jMax ) { idx1=lvs.j1(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoY=Min(infoY,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }
	if( k !=0     ) { idx1=lvs.k0(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoZ=Min(infoZ,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }
	if( k !=_kMax ) { idx1=lvs.k1(idx0); if(_hasPhi(stt[idx1])){ infoExist=true; infoZ=Min(infoZ,Abs(lvs[idx1])); /*if(_vel){sum+=vel[idx1];++cnt;}*/ } }

	if( !infoExist ) { return; }

	// update
	stt[idx0] = FMMState::updated;
	const float lvsEst = _solvePhi( infoX, infoY, infoZ );
	_updatePhi( lvs[idx0], sign*lvsEst );
	//if( cnt > 0 ) { vel[idx0] = sum * (1/(float)cnt); }

	// narrow band checking
	float& l = lvs[idx0];
	if( lvs[idx0] > 0 ) { if( l >  _posRange ) { return; } }
	else                { if( l < -_negRange ) { return; } }

	// marching
	if( i != 0 )
	{
		idx1 = lvs.i0(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i-1,j,k), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i-1,j,k), lvs[idx0]-1 ) ); }
		}
	}

	if( i!=_iMax )
	{
		idx1 = lvs.i1(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i+1,j,k), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i+1,j,k), lvs[idx0]-1 ) ); }
		}
	}

	if( j != 0 )
	{
		idx1 = lvs.j0(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j-1,k), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i,j-1,k), lvs[idx0]-1 ) ); }
		}
	}

	if( j!=_jMax )
	{
		idx1 = lvs.j1(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j+1,k), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i,j+1,k), lvs[idx0]-1 ) ); }
		}
	}

	if( k != 0 )
	{
		idx1 = lvs.k0(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j,k-1), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i,j,k-1), lvs[idx0]-1 ) ); }
		}
	}

	if( k!=_kMax )
	{
		idx1 = lvs.k1(idx0);

		if( stt[idx1] == FMMState::far )
		{
			stt[idx1] = FMMState::trial;

			if( sign > 0 ) { _posHeap.push( HEAPNODE( Vec3i(i,j,k+1), lvs[idx0]+1 ) ); }
			else           { _negHeap.push( HEAPNODE( Vec3i(i,j,k+1), lvs[idx0]-1 ) ); }
		}
	}
}

Vec3f
Voxelizer_FMM::_closestPointOnTriangle( const Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, Vec3f& baryCoords )
{
	const Vec3f AB(B-A), AC(C-A), PA(A-P);
	const float a=AB*AB, b=AB*AC, c=AC*AC, d=AB*PA, e=AC*PA;
	float det=a*c-b*b, s=b*e-c*d, t=b*d-a*e;

	if( s+t < det ) {
		if( s < 0.f ) {
			if( t < 0.f ) {
				if( d < 0.f ) { s=Clamp(-d/a,0.f,1.f); t=0.f; }
				else          { s=0.f; t=Clamp(-e/c,0.f,1.f); }
			} else {
				s=0.f; t=Clamp( -e/c, 0.f, 1.f );
			}
		} else if( t < 0.f ) {
			s=Clamp(-d/a,0.f,1.f); t=0.f;
		} else {
			det=1.f/det; s*=det; t*=det;
		}
	} else {
		if( s < 0.f ) {
			const float tmp0 = b+d;
			const float tmp1 = c+e;
			if( tmp1 > tmp0 ) {
				const float numer = tmp1 - tmp0;
				const float denom = a-2*b+c;
				s = Clamp( numer/denom, 0.f, 1.f );
				t = 1-s;
			} else {
				t = Clamp( -e/c, 0.f, 1.f );
				s = 0.f;
			}
		} else if( t < 0.f ) {
			if( a+d > b+e ) {
				const float numer = c+e-b-d;
				const float denom = a-2*b+c;
				s = Clamp( numer/denom, 0.f, 1.f );
				t = 1-s;
			} else {
				s = Clamp( -e/c, 0.f, 1.f );
				t = 0.f;
			}
		} else {
			const float numer = c+e-b-d;
			const float denom = a-2*b+c;
			s = Clamp( numer/denom, 0.f, 1.f );
			t = 1.f - s;
		}
	}

	baryCoords = Vec3f( (1-s-t), s, t );

	return Vec3f( (A.x+s*AB.x+t*AC.x), (A.y+s*AB.y+t*AC.y), (A.z+s*AB.z+t*AC.z) );
}

int 
Voxelizer_FMM::_baryCoords( Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, int whichPlane, Vec3f& baryCoords )
{
	float e = EPSILON;

	const float& x  = (whichPlane==0) ? P.x: ( (whichPlane==1) ? P.y : P.z );
	const float& y  = (whichPlane==0) ? P.y: ( (whichPlane==1) ? P.z : P.x );

	const float& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const float& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const float& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const float& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	const float& x2 = (whichPlane==0) ? C.x: ( (whichPlane==1) ? C.y : C.z );
	const float& y2 = (whichPlane==0) ? C.y: ( (whichPlane==1) ? C.z : C.x );

	const float denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);

	if( AlmostZero( denom, e ) )
	{
		baryCoords.zeroize();
		return -1;
	}

	const float& s = baryCoords[0] = ( (y1-y2)*(x-x2)+(x2-x1)*(y-y2) ) / denom;
	const float& t = baryCoords[1] = ( (y2-y0)*(x-x2)+(x0-x2)*(y-y2) ) / denom;
	baryCoords[2] = 1-s-t;

	return int( _isInsideTriangle( baryCoords ) );
}

bool
Voxelizer_FMM::_isInsideTriangle( const Vec3f& baryCoords )
{
	float e = EPSILON;
	const float& s = baryCoords[0];
	const float& t = baryCoords[1];

	if( s < -e ) { return false; }
	if( t < -e ) { return false; }
	if( (s+t) > (1+e) ) { return false; }

	return true;
}

Vec3f
Voxelizer_FMM::_normal( Vec3f& A, Vec3f& B, Vec3f& C )
{
	Vec3f N( (B-A)^(C-A) );
	N.normalize();
	return N;
}

bool
Voxelizer_FMM::_updatePhi( float& phi, float candidate )
{
	bool updated = false;
	if( Abs(phi) > Abs(candidate) ) { phi=candidate; updated=true; }
	return updated;
}

/**
 	Return true if the given state is 'FMMState::interface' or 'FMMState::updated' and false otherwise.
	@param[in] state The state being queried.
 	@return True if the given state is 'FMMState::interface' or 'FMMState::updated' and false otherwise.
*/
bool
Voxelizer_FMM::_hasPhi( int state )
{
	if( state==FMMState::updated   ) { return true; }
	if( state==FMMState::interface ) { return true; }
	return false;
}

/**
	Return the solution for the quadratic equation from 3 neighbors int x, y, and z-directions.
	The solution for the quadratic equation for 3 neighbors (xyz) is not critical.
	quadratic formula: (phi-phi_x)^2+(phi-phi_y)^2=1
	@note All the candidate values must not be negative.
	@param[in] p The candidate value in x-direction to update from it.
	@param[in] q The candidate value in y-direction to update from it.
	@param[in] r The candidate value in z-direction to update from it.
*/

float
Voxelizer_FMM::_solvePhi( float p, float q, float r )
{
	float d = Min(p,q,r) + 1; // 1: cell size
	if( d > Max(p,q,r) )
	{
		d = Min( d, 0.5f*((p+q)+sqrtf(2-Pow2(p-q))) );
		d = Min( d, 0.5f*((q+r)+sqrtf(2-Pow2(q-r))) );
		d = Min( d, 0.5f*((r+p)+sqrtf(2-Pow2(r-p))) );
	}
	return d;
}

std::ostream&
operator<<( std::ostream& os, const Voxelizer_FMM& solver )
{
	os << "<Voxelizer_FMM>" << std::endl;
	os << std::endl;

	return os;
}

BORA_NAMESPACE_END

