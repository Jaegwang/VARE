//----------------//
// Voxelizer_FSM.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2013.10.24                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Voxelizer_FSM::Voxelizer_FSM()
{
	reset();
}

void
Voxelizer_FSM::reset()
{
	_I0 = _I1 = _J0 = _J1 = _K0 = _K1 = 0;

	_initialState = true;
	_lvs       = (ScalarDenseField*)NULL;
	_stt       = (MarkerDenseField*)NULL;
	//_svl       = (ZVectorField*)NULL;

	_vPos      = (Vec3fArray*)NULL;
	//_vVel      = (ZVectorArray*)NULL;
	_triangles = (IndexArray*)NULL;
}

void
//Voxelizer_FSM::addMesh( ScalarDenseField& lvs, TriangleMesh& mesh, bool entireDomain )
Voxelizer_FSM::addMesh( ScalarDenseField& lvs, MarkerDenseField& stt, TriangleMesh& mesh, bool entireDomain )
{
	//if( !( lvs.location()==Z_NODE || lvs.location()==Z_CELL ) ) { return; }
	_lvs          = &lvs;
	_stt		  = &stt;
	//_svl          = (ZVectorField*)NULL;
	_vPos         = &mesh.pos;
	//_vVel         = (ZVectorArray*)NULL;
	_triangles    = &mesh.indices; 
	_entireDomain = entireDomain;

	_eps		  = 0.001f; //epsilon in voxel space

	compute();
}

//void
//Voxelizer_FSM::addMesh( ZScalarField& lvs, ZVectorField& svl, ZTriMesh& mesh, ZVectorArray& vVel, bool entireDomain )
//{
//	if( !lvs.directComputable( svl ) ) { return; }
//	if( mesh.numVertices() != vVel.length() ) { return; }
//	if( !( lvs.location()==Z_NODE || lvs.location()==Z_CELL ) ) { return; }
//
//	_lvs          = &lvs;
//	//_svl          = &svl;
//	_vPos         = &mesh.points;
//	//_vVel         = &vVel;
//	_triangles    = &mesh.triangles;
//	_entireDomain = entireDomain;
//
//	compute();
//}

void
Voxelizer_FSM::compute()
{
	ScalarDenseField& lvs = *_lvs;
	MarkerDenseField& stt = *_stt;
	_iMax = lvs.nx()-1;
	_jMax = lvs.ny()-1;
	_kMax = lvs.nz()-1;

	// set the computing region
	if( _entireDomain ) {

		_I0 = 0;   _I1 = _iMax;
		_J0 = 0;   _J1 = _jMax;
		_K0 = 0;   _K1 = _kMax;

	} 
//	else 
//	{
//
//		ZBoundingBox bBox( vPos.boundingBox() );
//		const ZPoint minPt( bBox.minPoint() ), maxPt( bBox.maxPoint() );
//
//		_I0 = ZClamp( ZTrunc(minPt.x), 0, _iMax );   _I1 = ZClamp( ZTrunc(maxPt.x)+1, 0, _iMax );
//		_J0 = ZClamp( ZTrunc(minPt.y), 0, _jMax );   _J1 = ZClamp( ZTrunc(maxPt.y)+1, 0, _jMax );
//		_K0 = ZClamp( ZTrunc(minPt.z), 0, _kMax );   _K1 = ZClamp( ZTrunc(maxPt.z)+1, 0, _kMax );
//
//		if( !((_I0<_I1)&&(_J0<_J1)&&(_K0<_K1))  ) { return; }
//	}

	_tagInterfacialElements();

	// six directions rather than eight directions
//	#pragma omp parallel sections
	{
//		#pragma omp section
		sweep( +1, +1, -1 );
//		#pragma omp section
		sweep( +1, +1, +1 );
//		#pragma omp section
		sweep( -1, +1, +1 );
//		#pragma omp section
		sweep( -1, +1, -1 );
//		#pragma omp section
		sweep( +1, -1, +1 );
//		#pragma omp section
		sweep( -1, -1, -1 );
	}
}

void
Voxelizer_FSM::_tagInterfacialElements()
{
	ScalarDenseField& 	lvs 		= *_lvs;
	MarkerDenseField& 	stt 		= *_stt;
	Vec3fArray&   		vPos		= *_vPos;
	IndexArray& 		triangles 	= *_triangles;

//	const bool hasVel = ( _vel && _vVel );
//	ZVector* vel  = hasVel ? (ZVector*)(_vel->pointer())  : (ZVector*)NULL;
//	ZVector* vVel = hasVel ? (ZVector*)(_vVel->pointer()) : (ZVector*)NULL;
	const int numVertices = vPos.capacity();
	const int numTrifaces = triangles.length()/3;

	if( !numVertices || !numTrifaces ) { return; }

	if( _initialState )
	{
		if( _lvs ) { lvs.setValueAll( _iMax+_jMax+_kMax ); }
		//if( _vel ) { _vel->zerorize();              }
		Grid lvsGrid = Grid( lvs.nx(), lvs.ny(), lvs.nz(), lvs.boundingBox() );
		stt.setGrid( lvsGrid ); stt.setValueAll( FSMState::far ); // TODO: change name from FSM to something more general	

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
	_dx = dx;
	_dy = dy;
	_dz = dz;

	const Vec3f minPt = aabb.minPoint(); // of boundingbox? get that first?
	const float ddx =1/dx,     ddy =1/dy,     ddz =1/dz;
	const float dx_2=0.5f*dx,  dy_2=0.5f*dy,  dz_2=0.5f*dz;

	for( int iTri=0; iTri<numTrifaces; iTri++ )
	{
		const int& v0 = triangles[iTri*3  ];
		const int& v1 = triangles[iTri*3+1];
		const int& v2 = triangles[iTri*3+2];
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
		// computing index range
//		i0 = Clamp( Floor(minCellInds.x), 0, _iMax ); i1 = Clamp( Ceil(maxCellInds.x), 0, _iMax );
//		j0 = Clamp( Floor(minCellInds.y), 0, _jMax ); j1 = Clamp( Ceil(maxCellInds.y), 0, _jMax );
//		k0 = Clamp( Floor(minCellInds.z), 0, _kMax ); k1 = Clamp( Ceil(maxCellInds.z), 0, _kMax );

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
							lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (i=I+1) <= _iMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (i=I+2) <= _iMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (i=(I-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(i*dx-x)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
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
							lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (j=J+1) <= _jMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (j=J+2) <= _jMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (j=(J-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(j*dy-y)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
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
							lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
						}

						if( (k=K+1) <= _kMax )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
							}

							if( (k=K+2) <= _kMax )
							{
								if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
								{
									lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
								}
							}
						}

						if( (k=(K-1)) >= 0 )
						{
							if( Abs(lvs[idx=lvs.cellIndex(i,j,k)]) > Abs(lvsEst=sgn*(k*dz-z)) )
							{
								lvs[idx]=lvsEst; stt[idx]=FSMState::interface; //if(hasVel){vel[idx]=velEst;}
							}
						}
					}
				}
			}}
		}
	}
}

void
Voxelizer_FSM::sweep( int di, int dj, int dk )
{
	int i0, i1; if(di>0) { i0=_I0+1; i1=_I1+1; } else { i0=_I1-1; i1=_I0-1; }
	int j0, j1; if(dj>0) { j0=_J0+1; j1=_J1+1; } else { j0=_J1-1; j1=_J0-1; }
	int k0, k1; if(dk>0) { k0=_K0+1; k1=_K1+1; } else { k0=_K1-1; k1=_K0-1; }

	for( int k=k0; k!=k1; k+=dk )
	for( int j=j0; j!=j1; j+=dj )
	for( int i=i0; i!=i1; i+=di )
	{
		update( i, j, k );
	}
}

void
Voxelizer_FSM::update( int i0, int j0, int k0 )
{

	ScalarDenseField& lvs = *_lvs;
	MarkerDenseField& stt = *_stt;
	//ZVectorField3D& vel = *_vel;

	const int &i=i0, &j=j0, &k=k0;
	const int idx0 = lvs.cellIndex( i, j, k );
	if( stt[idx0] == FSMState::interface ) { return; }

	bool infoExist = false;
	float infoX=BORA_LARGE, infoY=BORA_LARGE, infoZ=BORA_LARGE;

	int idx1, cnt=0;
	Vec3f sum;
	int signX, signY, signZ;
	if( i !=0     ) { idx1=lvs.i0(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoX>Abs(LVS_1)){infoX=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( i !=_iMax ) { idx1=lvs.i1(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoX>Abs(LVS_1)){infoX=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( j !=0     ) { idx1=lvs.j0(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoY>Abs(LVS_1)){infoY=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( j !=_jMax ) { idx1=lvs.j1(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoY>Abs(LVS_1)){infoY=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( k !=0     ) { idx1=lvs.k0(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoZ>Abs(LVS_1)){infoZ=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( k !=_kMax ) { idx1=lvs.k1(idx0); if(_hasPhi(stt[idx1]))
	{ infoExist=true; float LVS_1=lvs[idx1]; if(infoZ>Abs(LVS_1)){infoZ=LVS_1;} /*if(_vel){sum+=vel[idx1];++cnt;}*/ }}
	if( !infoExist ) { return; }

	// update
	stt[idx0] = FSMState::updated;
	const float lvsEst = _solvePhi( infoX, infoY, infoZ );
	_updatePhi( lvs[idx0], lvsEst );
	//if( cnt > 0 ) { vel[idx0] = sum * (1/(float)cnt); }

}

int 
Voxelizer_FSM::_baryCoords( Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, int whichPlane, Vec3f& baryCoords )
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
Voxelizer_FSM::_isInsideTriangle( const Vec3f& baryCoords )
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
Voxelizer_FSM::_normal( const Vec3f& A, const Vec3f& B, const Vec3f& C )
{
	Vec3f N( (B-A)^(C-A) );
	N.normalize();
	return N;
}

bool
Voxelizer_FSM::_updatePhi( float& phi, float candidate )
{
	bool updated = false;
	if( Abs(phi) > Abs(candidate) ) { phi=candidate; updated=true; }
	return updated;
}

/**
 	Return true if the given state is 'FSMState::interface' or 'FSMState::updated' and false otherwise.
	@param[in] state The state being queried.
 	@return True if the given state is 'FSMState::interface' or 'FSMState::updated' and false otherwise.
*/
bool
Voxelizer_FSM::_hasPhi( int state )
{
	if( state==FSMState::updated   ) { return true; }
	if( state==FSMState::interface ) { return true; }
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
Voxelizer_FSM::_solvePhi( float p, float q, float r )
{
	int sign = 1;
	int signX, signY, signZ;
	signX = signY = signZ = 1;
	if( p<0 ) { signX = -1; }
	if( q<0 ) { signY = -1; }
	if( r<0 ) { signZ = -1; }
	p = Abs(p); q = Abs(q); r= Abs(r);
	//float d = Min(p,q,r) + 1; // 1: cell size
	float d;
	if( p<q )
	{
		if( p<r ) { d = ((float)signX)*(p+1.f); }
		else	  { d = ((float)signZ)*(r+1.f); }
	}
	else //q<p
	{
		if( q<r ) { d = ((float)signY)*(q+1.f); }
		else	  { d = ((float)signZ)*(r+1.f); }
	}
	if( Abs(d) > Max(p,q,r) )
	{
		float pq = 0.5f*((p+q)+sqrtf(2-Pow2(p-q)));
		float qr = 0.5f*((q+r)+sqrtf(2-Pow2(q-r)));
		float rp = 0.5f*((r+p)+sqrtf(2-Pow2(r-p)));
		if( d>pq ) { d = pq; if( signX!=signY){sign=1;} else{sign=signX;} }
		if( d>qr ) { d = qr; if( signY!=signZ){sign=1;} else{sign=signY;} }
		if( d>rp ) { d = rp; if( signZ!=signX){sign=1;} else{sign=signZ;} }
	}
	return ((float)sign)*d;
}

std::ostream&
operator<<( std::ostream& os, const Voxelizer_FSM& solver )
{
	os << "<Voxelizer_FSM>" << std::endl;
	os << std::endl;

	return os;
}

BORA_NAMESPACE_END

