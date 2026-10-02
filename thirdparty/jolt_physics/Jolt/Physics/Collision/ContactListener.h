// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/Shape/SubShapeIDPair.h>
#include <Jolt/Core/StaticArray.h>

JPH_NAMESPACE_BEGIN

class Body;
class CollideShapeResult;

/// Array of contact points
using ContactPoints = StaticArray<Vec3, 64>;

/// Manifold class, describes the contact surface between two bodies
class ContactManifold
{
public:
	/// Swaps shape 1 and 2
	ContactManifold			SwapShapes() const					{ return { mBaseOffset, -mWorldSpaceNormal, mPenetrationDepth, mSubShapeID2, mSubShapeID1, mRelativeContactPointsOn2, mRelativeContactPointsOn1 }; }

	/// Access to the world space contact positions
	inline RVec3			GetWorldSpaceContactPointOn1(uint inIndex) const { return mBaseOffset + mRelativeContactPointsOn1[inIndex]; }
	inline RVec3			GetWorldSpaceContactPointOn2(uint inIndex) const { return mBaseOffset + mRelativeContactPointsOn2[inIndex]; }

	RVec3					mBaseOffset;						///< Offset to which all the contact points are relative
	Vec3					mWorldSpaceNormal;					///< Normal for this manifold, direction along which to move body 2 out of collision along the shortest path
	float					mPenetrationDepth;					///< Penetration depth (move shape 2 by this distance to resolve the collision). If this value is negative, this is a speculative contact point and may not actually result in a velocity change as during solving the bodies may not actually collide.
	SubShapeID				mSubShapeID1;						///< Sub shapes that formed this manifold (note that when multiple manifolds are combined because they're coplanar, we lose some information here because we only keep track of one sub shape pair that we encounter, see description at Body::SetUseManifoldReduction)
	SubShapeID				mSubShapeID2;
	ContactPoints			mRelativeContactPointsOn1;			///< Contact points on the surface of shape 1 relative to mBaseOffset.
	ContactPoints			mRelativeContactPointsOn2;			///< Contact points on the surface of shape 2 relative to mBaseOffset. If there's no penetration, this will be the same as mRelativeContactPointsOn1. If there is penetration they will be different.
};

// CRUMB: a compliant (soft) contact's state carried from one step to the next, stored after the contact
// points of the manifold's cache entry. It is reserved only for a manifold of a body with compliant
// shapes (Body::HasCompliantShapes when the entry was made). Vectors are in body 2's local frame, as the
// cache keeps the contact normal, so they keep their meaning when the next step picks other axes.
struct SoftContactTail
{
	Float3					mFrictionImpulse;					// The friction rows' total impulse on body 2
	Float3					mMoment;							// The tilt, roll and spin rows' total moment on body 2, about the patch centre
	Float3					mShear;								// The pre-slip deflection: body 2's contact surface against body 1's
	float					mNormalImpulse;						// The normal row's total impulse
	float					mCoreImpulse;						// The core guard's total impulse
	float					mSpinDeflection;					// The pre-slip twist about the normal (rad)
	float					mEndSquash;							// The deepest overlap the solve left for the end of the step
	uint32					mFlags;								// See EFlags
	uint32					mActiveSet[6];						// A face patch's contact set, one bit per grid cell, for the next step's first pass
	uint8					mGridLong;							// The grid the contact set belongs to
	uint8					mGridShort;
	uint16					mReserved;

	enum EFlags : uint32
	{
		OnsetPending		= 1 << 0,							// The episode's first touch is still to be decided (set when the manifold is new)
		Solved				= 1 << 1,							// A soft solve has written this tail
	};
};

// CRUMB: a compliant patch for one manifold, built by the contact listener (OnContactAdded and
// OnContactPersisted) through ContactSettings::mSoftPatch. When enabled, the solver replaces the
// manifold's contact points by these rows: a normal row at the centre of pressure that applies the
// patch law's force explicitly plus an implicit share, a core guard, tilt rows, a friction block (two
// tangents and the spin) limited by the soft-finger ellipse, and rolling resistance rows. All values are
// in the solver's units.
struct SoftContactPatch
{
	bool					mEnabled = false;					// The listener built a patch; otherwise the manifold stays rigid
	uint8					mClass = 0;							// 0 face, 1 line, 2 point
	bool					mRigidStick = false;				// The friction rows stick rigidly (no pre-slip springs)
	uint8					mNumTilt = 0;						// Rows about in-plane axes that carry tilt stiffness
	uint8					mNumRoll = 0;						// Rows about in-plane axes that carry rolling resistance
	RVec3					mCentre;							// Where the normal, friction and spin rows act (world); the deepest point when nothing presses
	RVec3					mDeepest;							// The deepest point of the overlap, midway between the two surfaces (world)
	Vec3					mTangent1 = Vec3::sZero();			// The patch's in-plane axes (world), tangent 2 = normal x tangent 1
	Vec3					mTangent2 = Vec3::sZero();
	float					mSquash = 0.0f;						// The overlap at mDeepest at the start of the step (negative: a gap)
	float					mForce = 0.0f;						// The law's force at the start of the step, never negative
	float					mNormalStiffness = 0.0f;			// d force / d squash at the start of the step
	float					mLawCoefficient = 0.0f;				// F(s) = C s^n from first touch, for a first touch's predicted peak
	float					mLawExponent = 1.0f;
	float					mNormalDampingRatio = 0.0f;			// Damping ratio of the normal row (the law's exponent)
	float					mDampingRatio = 0.0f;				// Damping ratio of the tilt, shear and spin rows (exponent 1)
	float					mRestitution = 0.0f;				// The material's restitution, for a first touch the step cannot resolve
	Vec3					mTiltAxis[2] = { Vec3::sZero(), Vec3::sZero() };
	float					mTiltStiffness[2] = { 0.0f, 0.0f };
	float					mTiltArmNeg[2] = { 0.0f, 0.0f };	// The moment about a tilt axis stays within the normal impulse times these
	float					mTiltArmPos[2] = { 0.0f, 0.0f };
	Vec3					mRollAxis[2] = { Vec3::sZero(), Vec3::sZero() };
	float					mRollArm[2] = { 0.0f, 0.0f };		// Rolling resistance arm about each roll axis
	float					mSpinRadius = 0.0f;					// The soft-finger ellipse's spin radius
	float					mStaticFriction = 0.0f;
	float					mSlidingFriction = 0.0f;
	float					mShearStiffness = 0.0f;				// Pre-slip springs, per tangent and about the normal (0 with mRigidStick)
	float					mSpinStiffness = 0.0f;
	float					mRelaxationLength = 0.0f;			// The pre-slip deflection relaxes over this much rolling
	float					mRollingRadius = 0.0f;				// Turns a relative spin about an in-plane axis into rolling speed
	float					mArea = 0.0f;
	float					mCoreGap = 0.0f;					// Layer left at mDeepest before the cores meet
	uint32					mActiveSet[6] = { 0, 0, 0, 0, 0, 0 };	// A face patch's contact set, carried to the next step
	uint8					mGridLong = 0;
	uint8					mGridShort = 0;
};

/// When a contact point is added or persisted, the callback gets a chance to override certain properties of the contact constraint.
/// The values are filled in with their defaults by the system so the callback doesn't need to modify anything, but it can if it wants to.
class ContactSettings
{
public:
	float					mCombinedFriction;					///< Combined friction for the body pair (see: PhysicsSystem::SetCombineFriction)
	float					mCombinedRestitution;				///< Combined restitution for the body pair (see: PhysicsSystem::SetCombineRestitution)
	float					mInvMassScale1 = 1.0f;				///< Scale factor for the inverse mass of body 1 (0 = infinite mass, 1 = use original mass, 2 = body has half the mass). For the same contact pair, you should strive to keep the value the same over time.
	float					mInvInertiaScale1 = 1.0f;			///< Scale factor for the inverse inertia of body 1 (usually same as mInvMassScale1)
	float					mInvMassScale2 = 1.0f;				///< Scale factor for the inverse mass of body 2 (0 = infinite mass, 1 = use original mass, 2 = body has half the mass). For the same contact pair, you should strive to keep the value the same over time.
	float					mInvInertiaScale2 = 1.0f;			///< Scale factor for the inverse inertia of body 2 (usually same as mInvMassScale2)
	bool					mIsSensor;							///< If the contact should be treated as a sensor vs body contact (no collision response)
	Vec3					mRelativeLinearSurfaceVelocity = Vec3::sZero(); ///< Relative linear surface velocity between the bodies (world space surface velocity of body 2 - world space surface velocity of body 1), can be used to create a conveyor belt effect
	Vec3					mRelativeAngularSurfaceVelocity = Vec3::sZero(); ///< Relative angular surface velocity between the bodies (world space angular surface velocity of body 2 - world space angular surface velocity of body 1). Note that this angular velocity is relative to the center of mass of body 1, so if you want it relative to body 2's center of mass you need to add body 2 angular velocity x (body 1 world space center of mass - body 2 world space center of mass) to mRelativeLinearSurfaceVelocity.
	// CRUMB: ask for ContactListener::OnContactSolved on this manifold after the velocity solve, and
	// on a hit the sweep path (continuous collision detection) resolves with these settings. It
	// changes nothing the solver does.
	bool					mReportSolved = false;
	// CRUMB: the compliant contact. For a manifold of a body with compliant shapes the contact manager
	// points mSoftPatch at a record the listener may fill (nullptr: this manifold cannot carry a patch)
	// and mSoftTail at the manifold's tail from the last step (nullptr: a new episode). mIsSweepPath is
	// set when the call comes from the sweep path (continuous collision detection), whose poses are not
	// the time of impact: no patch is built there.
	SoftContactPatch *		mSoftPatch = nullptr;
	const SoftContactTail *	mSoftTail = nullptr;
	bool					mIsSweepPath = false;
};

// CRUMB: what the solver applied to one manifold over the step, on body 2 (body 1 takes the opposite).
// Built after the velocity iterations from the total impulses, so it is what the bodies felt, not an
// estimate made before the solve. Positions are not integrated yet when it is built, so the bodies
// are where the solve saw them. A hit the sweep path resolves is reported the same way, right after
// it applies its impulses: one point, at the hit, with mBody1Offset saying how far into the step the
// swept body (body 1) had travelled when it was struck.
struct ContactSolvedReport
{
	Vec3					mWorldNormal;						// From body 1 to body 2
	RVec3					mCentre;							// Normal-impulse-weighted contact point (world); the plain mean of the points when nothing pressed
	Vec3					mLinearImpulse;						// Total impulse on body 2, normal plus friction (world)
	Vec3					mAngularImpulse;					// Moment of the impulses about mCentre on body 2 (world), twist and tilt included
	float					mNormalImpulse;						// Sum of the pressing impulses, never negative
	float					mArea;								// A soft patch's contact area, 0 for a rigid manifold
	float					mFrictionUse;						// 0 to 1: the friction impulses' magnitudes over their limit, summed over the points (1 = sliding)
	Vec3					mBody1Offset = Vec3::sZero();		// Body 1's centre of mass when the impulses were applied, less where it is now: zero after the velocity solve, the travel to the hit on the sweep path
};

/// Return value for the OnContactValidate callback. Determines if the contact is being processed or not.
/// Results are ordered so that the strongest accept has the lowest number and the strongest reject the highest number (which allows for easy combining of results)
enum class ValidateResult
{
	AcceptAllContactsForThisBodyPair,							///< Accept this and any further contact points for this body pair
	AcceptContact,												///< Accept this contact only (and continue calling this callback for every contact manifold for the same body pair)
	RejectContact,												///< Reject this contact only (but process any other contact manifolds for the same body pair)
	RejectAllContactsForThisBodyPair							///< Rejects this and any further contact points for this body pair
};

/// A listener class that receives collision contact events. It can be registered through PhysicsSystem::SetContactListener.
/// Only a single contact listener can be registered. A common pattern is to create a contact listener that casts Body::GetUserData
/// to a game object and then forwards the call to a handler specific for that game object.
/// Typically this is done on both objects involved in a collision event.
///
/// Note that contact listener callbacks are called from multiple threads at the same time when all bodies are locked, this means you cannot
/// use PhysicsSystem::GetBodyInterface / PhysicsSystem::GetBodyLockInterface but must use PhysicsSystem::GetBodyInterfaceNoLock / PhysicsSystem::GetBodyLockInterfaceNoLock instead.
/// If you use a locking interface, the simulation will deadlock. You're only allowed to read from the bodies and you can't change physics state.
/// During OnContactRemoved you cannot access the bodies at all, see the comments at that function.
///
/// While a callback can come from multiple threads, all callbacks relating to a single body pair are serialized.
/// For EMotionQuality::Discrete bodies, during every 'collision step' in a PhysicsSystem::Update, you will receive at most one OnContactAdded/Persisted/Removed call per body/sub shape pair.
/// For EMotionQuality::LinearCast bodies, you may get an OnContactAdded followed by an OnContactPersisted for the same body/sub shape pair.
/// This happens when a body collides both in the discrete and the continuous collision detection stage.
class ContactListener
{
public:
	/// Ensure virtual destructor
	virtual					~ContactListener() = default;

	/// Called after detecting a collision between a body pair, but before calling OnContactAdded and before adding the contact constraint.
	/// If the function rejects the contact, the contact will not be processed by the simulation.
	/// This is a rather expensive time to reject a contact point since a lot of the collision detection has happened already, make sure you
	/// filter out the majority of undesired body pairs through the ObjectLayerPairFilter that is registered on the PhysicsSystem.
	///
	/// This function may not be called again the next update if a contact persists and no new contact pairs between sub shapes are found.
	///
	/// Note that this callback is called when all bodies are locked, so don't use any locking functions! See detailed class description of ContactListener.
	///
	/// Body 1 will have a motion type that is larger or equal than body 2's motion type (order from large to small: dynamic -> kinematic -> static). When motion types are equal, they are ordered by BodyID.
	///
	/// The collision result (inCollisionResult) is reported relative to inBaseOffset.
	virtual ValidateResult	OnContactValidate([[maybe_unused]] const Body &inBody1, [[maybe_unused]] const Body &inBody2, [[maybe_unused]] RVec3Arg inBaseOffset, [[maybe_unused]] const CollideShapeResult &inCollisionResult) { return ValidateResult::AcceptAllContactsForThisBodyPair; }

	/// Called whenever a new contact point is detected.
	///
	/// Note that this callback is called when all bodies are locked, so don't use any locking functions! See detailed class description of ContactListener.
	///
	/// Body 1 and 2 will be sorted such that body 1 ID < body 2 ID, so body 1 may not be dynamic.
	///
	/// Note that only active bodies will report contacts, as soon as a body goes to sleep the contacts between that body and all other
	/// bodies will receive an OnContactRemoved callback, if this is the case then Body::IsActive() will return false during the callback.
	///
	/// When contacts are added, the constraint solver has not run yet, so the collision impulse is unknown at that point.
	/// The velocities of inBody1 and inBody2 are the velocities before the contact has been resolved, so you can use this to
	/// estimate the collision impulse to e.g. determine the volume of the impact sound to play (see: EstimateCollisionResponse).
	virtual void			OnContactAdded([[maybe_unused]] const Body &inBody1, [[maybe_unused]] const Body &inBody2, [[maybe_unused]] const ContactManifold &inManifold, [[maybe_unused]] ContactSettings &ioSettings) { /* Do nothing */ }

	/// Called whenever a contact is detected that was also detected last update.
	///
	/// Note that this callback is called when all bodies are locked, so don't use any locking functions! See detailed class description of ContactListener.
	///
	/// Body 1 and 2 will be sorted such that body 1 ID < body 2 ID, so body 1 may not be dynamic.
	///
	/// If the structure of the shape of a body changes between simulation steps (e.g. by adding/removing a child shape of a compound shape),
	/// it is possible that the same sub shape ID used to identify the removed child shape is now reused for a different child shape. The physics
	/// system cannot detect this, so may send a 'contact persisted' callback even though the contact is now on a different child shape. You can
	/// detect this by keeping the old shape (before adding/removing a part) around until the next PhysicsSystem::Update (when the OnContactPersisted
	/// callbacks are triggered) and resolving the sub shape ID against both the old and new shape to see if they still refer to the same child shape.
	virtual void			OnContactPersisted([[maybe_unused]] const Body &inBody1, [[maybe_unused]] const Body &inBody2, [[maybe_unused]] const ContactManifold &inManifold, [[maybe_unused]] ContactSettings &ioSettings) { /* Do nothing */ }

	/// Called whenever a contact was detected last update but is not detected anymore.
	///
	/// You cannot access the bodies at the time of this callback because:
	/// - All bodies are locked at the time of this callback.
	/// - Some properties of the bodies are being modified from another thread at the same time.
	/// - The body may have been removed and destroyed (you'll receive an OnContactRemoved callback in the PhysicsSystem::Update after the body has been removed).
	///
	/// Cache what you need in the OnContactAdded and OnContactPersisted callbacks and store it in a separate structure to use during this callback.
	/// Alternatively, you could just record that the contact was removed and process it after PhysicsSystem::Update.
	///
	/// Body 1 and 2 will be sorted such that body 1 ID < body 2 ID, so body 1 may not be dynamic.
	///
	/// The sub shape IDs were created in the previous simulation step, so if the structure of a shape changes (e.g. by adding/removing a child shape of a compound shape),
	/// the sub shape ID may not be valid / may not point to the same sub shape anymore.
	/// If you want to know if this is the last contact between the two bodies, use PhysicsSystem::WereBodiesInContact.
	virtual void			OnContactRemoved([[maybe_unused]] const SubShapeIDPair &inSubShapePair) { /* Do nothing */ }

	// CRUMB: called after the velocity solve for every manifold whose settings asked for it
	// (ContactSettings::mReportSolved), with what the solver applied over the step, and after every hit
	// the sweep path resolves for a contact whose settings asked for it, with what that hit applied.
	// Called from the solver with the bodies locked, like OnContactAdded: read only. Body 1 and 2 are
	// the constraint's; on the sweep path body 1 is the body that swept.
	virtual void			OnContactSolved([[maybe_unused]] const Body &inBody1, [[maybe_unused]] const Body &inBody2, [[maybe_unused]] const SubShapeIDPair &inSubShapePair, [[maybe_unused]] const ContactSolvedReport &inReport) { /* Do nothing */ }
};

JPH_NAMESPACE_END
