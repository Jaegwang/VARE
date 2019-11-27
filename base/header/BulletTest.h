//--------------//
// BulletTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.12                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>
#include <btBulletDynamicsCommon.h>

BORA_NAMESPACE_BEGIN

class BulletTest
{
    private:

        btDefaultCollisionConfiguration* collisionConfiguration = 0;

        btCollisionDispatcher* dispatcher = 0;

        btBroadphaseInterface* overlappingPairCache = 0;

        btSequentialImpulseConstraintSolver* solver = 0;

        btDiscreteDynamicsWorld* dynamicsWorld = 0;

		btCollisionShape* instanceShape = 0;

    	btAlignedObjectArray<btCollisionShape*> collisionShapes;

    public:

        std::vector< btVector3 >   origins;
        std::vector< btMatrix3x3 > basises;

    public:

        void initialize();

        void simulate();

        void clear();

        void addRigidBody( const Vec3f& pos, const Vec3f& vel );
};


inline void
BulletTest::addRigidBody( const Vec3f& pos, const Vec3f& vel )
{
    if( dynamicsWorld == 0 ) return;
    
    if( instanceShape == 0 ) instanceShape = new btSphereShape(btScalar(1.));

    /// Create Dynamic Objects
    btTransform startTransform;
    startTransform.setIdentity();

    btScalar mass(1.f);

    //rigidbody is dynamic if and only if mass is non zero, otherwise static
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0, 0, 0);
    if (isDynamic) instanceShape->calculateLocalInertia(mass, localInertia);

    startTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

    //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
    btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(mass, myMotionState, instanceShape, localInertia);

    btRigidBody* body = new btRigidBody(rbInfo);
    body->setLinearVelocity(btVector3(vel.x, vel.y, vel.z));

    dynamicsWorld->addRigidBody(body);
}

BORA_NAMESPACE_END

