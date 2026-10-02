// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/Constraints/CalculateSolverSteps.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsUpdateContext.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/IslandBuilder.h>
#include <Jolt/Physics/DeterminismLog.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/QuickSort.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

using namespace literals;

#ifdef JPH_DEBUG_RENDERER
bool ContactConstraintManager::sDrawContactPoint = false;
bool ContactConstraintManager::sDrawSupportingFaces = false;
bool ContactConstraintManager::sDrawContactPointReduction = false;
bool ContactConstraintManager::sDrawContactManifolds = false;
#endif // JPH_DEBUG_RENDERER

//#define JPH_MANIFOLD_CACHE_DEBUG

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::WorldContactPoint
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::WorldContactPoint::CalculateNonPenetrationConstraintProperties(const Body &inBody1, float inInvMass1, float inInvInertiaScale1, const Body &inBody2, float inInvMass2, float inInvInertiaScale2, RVec3Arg inWorldSpacePosition1, RVec3Arg inWorldSpacePosition2, Vec3Arg inWorldSpaceNormal)
{
	// Calculate collision points relative to body
	RVec3 p = 0.5_r * (inWorldSpacePosition1 + inWorldSpacePosition2);
	Vec3 r1 = Vec3(p - inBody1.GetCenterOfMassPosition());
	Vec3 r2 = Vec3(p - inBody2.GetCenterOfMassPosition());

	mNonPenetrationConstraint.CalculateConstraintPropertiesWithMassOverride(inBody1, inInvMass1, inInvInertiaScale1, r1, inBody2, inInvMass2, inInvInertiaScale2, r2, inWorldSpaceNormal);
}

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE void ContactConstraintManager::WorldContactPoint::TemplatedCalculateFrictionAndNonPenetrationConstraintProperties(float inDeltaTime, Vec3Arg inGravity, const Body &inBody1, const Body &inBody2, float inInvM1, float inInvM2, Mat44Arg inInvI1, Mat44Arg inInvI2, RVec3Arg inWorldSpacePosition1, RVec3Arg inWorldSpacePosition2, Vec3Arg inWorldSpaceNormal, Vec3Arg inWorldSpaceTangent1, Vec3Arg inWorldSpaceTangent2, const ContactSettings &inSettings, float inMinVelocityForRestitution)
{
	JPH_DET_LOG("TemplatedCalculateFrictionAndNonPenetrationConstraintProperties: p1: " << inWorldSpacePosition1 << " p2: " << inWorldSpacePosition2
		<< " normal: " << inWorldSpaceNormal << " tangent1: " << inWorldSpaceTangent1 << " tangent2: " << inWorldSpaceTangent2
		<< " restitution: " << inSettings.mCombinedRestitution << " friction: " << inSettings.mCombinedFriction << " minv: " << inMinVelocityForRestitution
		<< " surface_vel: " << inSettings.mRelativeLinearSurfaceVelocity << " surface_ang: " << inSettings.mRelativeAngularSurfaceVelocity);

	// Calculate collision points relative to body
	RVec3 p = 0.5_r * (inWorldSpacePosition1 + inWorldSpacePosition2);
	Vec3 r1 = Vec3(p - inBody1.GetCenterOfMassPosition());
	Vec3 r2 = Vec3(p - inBody2.GetCenterOfMassPosition());

	const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
	const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();

	// Calculate velocity of collision points
	Vec3 relative_velocity;
	if constexpr (Type1 != EMotionType::Static && Type2 != EMotionType::Static)
		relative_velocity = mp2->GetPointVelocityCOM(r2) - mp1->GetPointVelocityCOM(r1);
	else if constexpr (Type1 != EMotionType::Static)
		relative_velocity = -mp1->GetPointVelocityCOM(r1);
	else if constexpr (Type2 != EMotionType::Static)
		relative_velocity = mp2->GetPointVelocityCOM(r2);
	else
	{
		JPH_ASSERT(false, "Static vs static makes no sense");
		relative_velocity = Vec3::sZero();
	}
	float normal_velocity = relative_velocity.Dot(inWorldSpaceNormal);

	// How much the shapes are penetrating (> 0 if penetrating, < 0 if separated)
	float penetration = Vec3(inWorldSpacePosition1 - inWorldSpacePosition2).Dot(inWorldSpaceNormal);

	// If there is no penetration, this is a speculative contact and we will apply a bias to the contact constraint
	// so that the constraint becomes relative_velocity . contact normal > -penetration / delta_time
	// instead of relative_velocity . contact normal > 0
	// See: GDC 2013: "Physics for Game Programmers; Continuous Collision" - Erin Catto
	float speculative_contact_velocity_bias = max(0.0f, -penetration / inDeltaTime);

	// Determine if the velocity is big enough for restitution
	float normal_velocity_bias;
	if (inSettings.mCombinedRestitution > 0.0f && normal_velocity < -inMinVelocityForRestitution)
	{
		// We have a velocity that is big enough for restitution. This is where speculative contacts don't work
		// great as we have to decide now if we're going to apply the restitution or not. If the relative
		// velocity is big enough for a hit, we apply the restitution (in the end, due to other constraints,
		// the objects may actually not collide and we will have applied restitution incorrectly). Another
		// artifact that occurs because of this approximation is that the object will bounce from its current
		// position rather than from a position where it is touching the other object. This causes the object
		// to appear to move faster for 1 frame (the opposite of time stealing).
		if (normal_velocity < -speculative_contact_velocity_bias)
		{
			// The gravity / constant forces are applied in the beginning of the time step.
			// If we get here, there was a collision at the beginning of the time step, so we've applied too much force.
			// This means that our calculated restitution can be too high resulting in an increase in energy.
			// So, when we apply restitution, we cancel the added velocity due to these forces.
			Vec3 relative_acceleration;

			// Calculate effect of gravity
			if constexpr (Type1 != EMotionType::Static && Type2 != EMotionType::Static)
				relative_acceleration = inGravity * (mp2->GetGravityFactor() - mp1->GetGravityFactor());
			else if constexpr (Type1 != EMotionType::Static)
				relative_acceleration = -inGravity * mp1->GetGravityFactor();
			else if constexpr (Type2 != EMotionType::Static)
				relative_acceleration = inGravity * mp2->GetGravityFactor();
			else
			{
				JPH_ASSERT(false, "Static vs static makes no sense");
				relative_acceleration = Vec3::sZero();
			}

			// Calculate effect of accumulated forces
			if constexpr (Type1 == EMotionType::Dynamic)
				relative_acceleration -= mp1->GetAccumulatedForce() * mp1->GetInverseMass();
			if constexpr (Type2 == EMotionType::Dynamic)
				relative_acceleration += mp2->GetAccumulatedForce() * mp2->GetInverseMass();

			// We only compensate forces towards the contact normal.
			float force_delta_velocity = min(0.0f, relative_acceleration.Dot(inWorldSpaceNormal) * inDeltaTime);

			normal_velocity_bias = inSettings.mCombinedRestitution * (normal_velocity - force_delta_velocity);
		}
		else
		{
			// In this case we have predicted that we don't hit the other object, but if we do (due to other constraints changing velocities)
			// the speculative contact will prevent penetration but will not apply restitution leading to another artifact.
			normal_velocity_bias = speculative_contact_velocity_bias;
		}
	}
	else
	{
		// No restitution. We can safely apply our contact velocity bias.
		normal_velocity_bias = speculative_contact_velocity_bias;
	}

	mNonPenetrationConstraint.TemplatedCalculateConstraintProperties<Type1, Type2>(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceNormal, normal_velocity_bias);

	// Calculate friction part
	if (inSettings.mCombinedFriction > 0.0f)
	{
		// Get surface velocity relative to tangents
		Vec3 ws_surface_velocity = inSettings.mRelativeLinearSurfaceVelocity + inSettings.mRelativeAngularSurfaceVelocity.Cross(r1);
		float surface_velocity1 = inWorldSpaceTangent1.Dot(ws_surface_velocity);
		float surface_velocity2 = inWorldSpaceTangent2.Dot(ws_surface_velocity);

		// Implement friction as 2 AxisConstraintParts
		mFrictionConstraint1.TemplatedCalculateConstraintProperties<Type1, Type2>(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceTangent1, surface_velocity1);
		mFrictionConstraint2.TemplatedCalculateConstraintProperties<Type1, Type2>(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceTangent2, surface_velocity2);
	}
	else
	{
		// Turn off friction constraint
		mFrictionConstraint1.Deactivate();
		mFrictionConstraint2.Deactivate();
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::ContactConstraint
////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef JPH_DEBUG_RENDERER
void ContactConstraintManager::ContactConstraint::Draw(DebugRenderer *inRenderer, ColorArg inManifoldColor) const
{
	if (mContactPoints.empty())
		return;

	// Get body transforms
	RMat44 transform_body1 = mBody1->GetCenterOfMassTransform();
	RMat44 transform_body2 = mBody2->GetCenterOfMassTransform();

	RVec3 prev_point = transform_body1 * Vec3::sLoadFloat3Unsafe(mContactPoints.back().mContactPoint->mPosition1);
	for (const WorldContactPoint &wcp : mContactPoints)
	{
		// Test if any lambda from the previous frame was transferred
		float radius = wcp.mNonPenetrationConstraint.GetTotalLambda() == 0.0f
					&& wcp.mFrictionConstraint1.GetTotalLambda() == 0.0f
					&& wcp.mFrictionConstraint2.GetTotalLambda() == 0.0f? 0.1f :  0.2f;

		RVec3 next_point = transform_body1 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition1);
		inRenderer->DrawMarker(next_point, Color::sCyan, radius);
		inRenderer->DrawMarker(transform_body2 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition2), Color::sPurple, radius);

		// Draw edge
		inRenderer->DrawArrow(prev_point, next_point, inManifoldColor, 0.05f);
		prev_point = next_point;
	}

	// Draw normal
	RVec3 wp = transform_body1 * Vec3::sLoadFloat3Unsafe(mContactPoints[0].mContactPoint->mPosition1);
	inRenderer->DrawArrow(wp, wp + GetWorldSpaceNormal(), Color::sRed, 0.05f);

	// Get tangents
	Vec3 t1, t2;
	GetTangents(t1, t2);

	// Draw tangents
	inRenderer->DrawLine(wp, wp + t1, Color::sGreen);
	inRenderer->DrawLine(wp, wp + t2, Color::sBlue);
}
#endif // JPH_DEBUG_RENDERER

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedContactPoint
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedContactPoint::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mPosition1);
	inStream.Write(mPosition2);
	inStream.Write(mNonPenetrationLambda);
	inStream.Write(mFrictionLambda);
}

void ContactConstraintManager::CachedContactPoint::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mPosition1);
	inStream.Read(mPosition2);
	inStream.Read(mNonPenetrationLambda);
	inStream.Read(mFrictionLambda);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedManifold
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedManifold::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mContactNormal);
}

void ContactConstraintManager::CachedManifold::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mContactNormal);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedBodyPair
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedBodyPair::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mDeltaPosition);
	inStream.Write(mDeltaRotation);
}

void ContactConstraintManager::CachedBodyPair::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mDeltaPosition);
	inStream.Read(mDeltaRotation);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::ManifoldCache
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::ManifoldCache::Init(uint inMaxBodyPairs, uint inMaxContactConstraints, uint inCachedManifoldsSize)
{
	uint max_body_pairs = min(inMaxBodyPairs, cMaxBodyPairsLimit);
	JPH_ASSERT(max_body_pairs == inMaxBodyPairs, "Cannot support this many body pairs!");
	JPH_ASSERT(inMaxContactConstraints <= cMaxContactConstraintsLimit); // Should have been enforced by caller

	mAllocator.Init(uint(min(uint64(max_body_pairs) * sizeof(BodyPairMap::KeyValue) + inCachedManifoldsSize, uint64(~uint(0)))));

	mCachedManifolds.Init(GetNextPowerOf2(inMaxContactConstraints));
	mCachedBodyPairs.Init(GetNextPowerOf2(max_body_pairs));
}

void ContactConstraintManager::ManifoldCache::Clear()
{
	JPH_PROFILE_FUNCTION();

	mCachedManifolds.Clear();
	mCachedBodyPairs.Clear();
	mAllocator.Clear();

#ifdef JPH_ENABLE_ASSERTS
	// Mark as incomplete
	mIsFinalized = false;
#endif
}

void ContactConstraintManager::ManifoldCache::Prepare(uint inExpectedNumBodyPairs, uint inExpectedNumManifolds)
{
	// Minimum amount of buckets to use in the hash map
	constexpr uint32 cMinBuckets = 1024;

	// Use the next higher power of 2 of amount of objects in the cache from last frame to determine the amount of buckets in this frame
	mCachedManifolds.SetNumBuckets(min(max(cMinBuckets, GetNextPowerOf2(inExpectedNumManifolds)), mCachedManifolds.GetMaxBuckets()));
	mCachedBodyPairs.SetNumBuckets(min(max(cMinBuckets, GetNextPowerOf2(inExpectedNumBodyPairs)), mCachedBodyPairs.GetMaxBuckets()));
}

const ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::Find(const SubShapeIDPair &inKey, uint64 inKeyHash) const
{
	JPH_ASSERT(mIsFinalized);
	return mCachedManifolds.Find(inKey, inKeyHash);
}

ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::Create(ContactAllocator &ioContactAllocator, const SubShapeIDPair &inKey, uint64 inKeyHash, int inNumContactPoints, bool inSoftTail)
{
	JPH_ASSERT(!mIsFinalized);
	MKeyValue *kv = mCachedManifolds.Create(ioContactAllocator, inKey, inKeyHash, CachedManifold::sGetRequiredExtraSize(inNumContactPoints, inSoftTail));
	if (kv == nullptr)
	{
		ioContactAllocator.mErrors |= EPhysicsUpdateError::ManifoldCacheFull;
		return nullptr;
	}
	kv->GetValue().mNumContactPoints = uint16(inNumContactPoints);
	if (inSoftTail)
		kv->GetValue().mFlags.fetch_or(uint16(CachedManifold::EFlags::SoftTail), memory_order_relaxed); // CRUMB
	++ioContactAllocator.mNumManifolds;
	return kv;
}

ContactConstraintManager::MKVAndCreated ContactConstraintManager::ManifoldCache::FindOrCreate(ContactAllocator &ioContactAllocator, const SubShapeIDPair &inKey, uint64 inKeyHash, int inNumContactPoints)
{
	MKeyValue *kv = const_cast<MKeyValue *>(mCachedManifolds.Find(inKey, inKeyHash));
	if (kv != nullptr)
		return { kv, false };

	return { Create(ioContactAllocator, inKey, inKeyHash, inNumContactPoints), true };
}

uint32 ContactConstraintManager::ManifoldCache::ToHandle(const MKeyValue *inKeyValue) const
{
	JPH_ASSERT(!mIsFinalized);
	return mCachedManifolds.ToHandle(inKeyValue);
}

const ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::FromHandle(uint32 inHandle) const
{
	JPH_ASSERT(mIsFinalized);
	return mCachedManifolds.FromHandle(inHandle);
}

const ContactConstraintManager::BPKeyValue *ContactConstraintManager::ManifoldCache::Find(const BodyPair &inKey, uint64 inKeyHash) const
{
	JPH_ASSERT(mIsFinalized);
	return mCachedBodyPairs.Find(inKey, inKeyHash);
}

ContactConstraintManager::BPKeyValue *ContactConstraintManager::ManifoldCache::Create(ContactAllocator &ioContactAllocator, const BodyPair &inKey, uint64 inKeyHash)
{
	JPH_ASSERT(!mIsFinalized);
	BPKeyValue *kv = mCachedBodyPairs.Create(ioContactAllocator, inKey, inKeyHash, 0);
	if (kv == nullptr)
	{
		ioContactAllocator.mErrors |= EPhysicsUpdateError::BodyPairCacheFull;
		return nullptr;
	}
	++ioContactAllocator.mNumBodyPairs;
	return kv;
}

void ContactConstraintManager::ManifoldCache::GetAllBodyPairsSorted(Array<const BPKeyValue *> &outAll) const
{
	JPH_ASSERT(mIsFinalized);
	mCachedBodyPairs.GetAllKeyValues(outAll);

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const BPKeyValue *inLHS, const BPKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::GetAllManifoldsSorted(const CachedBodyPair &inBodyPair, Array<const MKeyValue *> &outAll) const
{
	JPH_ASSERT(mIsFinalized);

	// Iterate through the attached manifolds
	for (uint32 handle = inBodyPair.mFirstCachedManifold; handle != ManifoldMap::cInvalidHandle; handle = FromHandle(handle)->GetValue().mNextWithSameBodyPair)
	{
		const MKeyValue *kv = mCachedManifolds.FromHandle(handle);
		outAll.push_back(kv);
	}

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const MKeyValue *inLHS, const MKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::GetAllCCDManifoldsSorted(Array<const MKeyValue *> &outAll) const
{
	mCachedManifolds.GetAllKeyValues(outAll);

	for (int i = (int)outAll.size() - 1; i >= 0; --i)
		if ((outAll[i]->GetValue().mFlags & (uint16)CachedManifold::EFlags::CCDContact) == 0)
		{
			outAll[i] = outAll.back();
			outAll.pop_back();
		}

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const MKeyValue *inLHS, const MKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::ContactPointRemovedCallbacks(ContactListener *inListener)
{
	JPH_PROFILE_FUNCTION();

	for (MKeyValue &kv : mCachedManifolds)
		if ((kv.GetValue().mFlags & uint16(CachedManifold::EFlags::ContactPersisted)) == 0)
			inListener->OnContactRemoved(kv.GetKey());
}

#ifdef JPH_ENABLE_ASSERTS

void ContactConstraintManager::ManifoldCache::Finalize()
{
	mIsFinalized = true;

#ifdef JPH_MANIFOLD_CACHE_DEBUG
	Trace("ManifoldMap:");
	mCachedManifolds.TraceStats();
	Trace("BodyPairMap:");
	mCachedBodyPairs.TraceStats();
#endif // JPH_MANIFOLD_CACHE_DEBUG
}

#endif

void ContactConstraintManager::ManifoldCache::SaveState(StateRecorder &inStream, const StateRecorderFilter *inFilter) const
{
	JPH_ASSERT(mIsFinalized);

	// Get contents of cache
	Array<const BPKeyValue *> all_bp;
	GetAllBodyPairsSorted(all_bp);

	// Determine which ones to save
	Array<const BPKeyValue *> selected_bp;
	if (inFilter == nullptr)
		selected_bp = std::move(all_bp);
	else
	{
		selected_bp.reserve(all_bp.size());
		for (const BPKeyValue *bp_kv : all_bp)
			if (inFilter->ShouldSaveContact(bp_kv->GetKey().mBodyA, bp_kv->GetKey().mBodyB))
				selected_bp.push_back(bp_kv);
	}

	// Write body pairs
	uint32 num_body_pairs = uint32(selected_bp.size());
	inStream.Write(num_body_pairs);
	for (const BPKeyValue *bp_kv : selected_bp)
	{
		// Write body pair key
		inStream.Write(bp_kv->GetKey());

		// Write body pair
		const CachedBodyPair &bp = bp_kv->GetValue();
		bp.SaveState(inStream);

		// Get attached manifolds
		Array<const MKeyValue *> all_m;
		GetAllManifoldsSorted(bp, all_m);

		// Write num manifolds
		uint32 num_manifolds = uint32(all_m.size());
		inStream.Write(num_manifolds);

		// Write all manifolds
		for (const MKeyValue *m_kv : all_m)
		{
			// Write key
			inStream.Write(m_kv->GetKey());
			const CachedManifold &cm = m_kv->GetValue();
			JPH_ASSERT((cm.mFlags & (uint16)CachedManifold::EFlags::CCDContact) == 0);

			// Write amount of contacts
			inStream.Write(cm.mNumContactPoints);

			// CRUMB: whether a soft-contact tail follows the points
			bool soft_tail = cm.HasSoftTail();
			inStream.Write(soft_tail);

			// Write manifold
			cm.SaveState(inStream);

			// Write contact points
			for (uint32 i = 0; i < cm.mNumContactPoints; ++i)
				cm.mContactPoints[i].SaveState(inStream);

			// CRUMB: and the tail
			if (soft_tail)
				inStream.Write(*cm.GetSoftTail());
		}
	}

	// Get CCD manifolds
	Array<const MKeyValue *> all_m;
	GetAllCCDManifoldsSorted(all_m);

	// Determine which ones to save
	Array<const MKeyValue *> selected_m;
	if (inFilter == nullptr)
		selected_m = std::move(all_m);
	else
	{
		selected_m.reserve(all_m.size());
		for (const MKeyValue *m_kv : all_m)
			if (inFilter->ShouldSaveContact(m_kv->GetKey().GetBody1ID(), m_kv->GetKey().GetBody2ID()))
				selected_m.push_back(m_kv);
	}

	// Write all CCD manifold keys
	uint32 num_manifolds = uint32(selected_m.size());
	inStream.Write(num_manifolds);
	for (const MKeyValue *m_kv : selected_m)
		inStream.Write(m_kv->GetKey());
}

bool ContactConstraintManager::ManifoldCache::RestoreState(const ManifoldCache &inReadCache, StateRecorder &inStream, const StateRecorderFilter *inFilter)
{
	JPH_ASSERT(!mIsFinalized);

	bool success = true;

	// Create a contact allocator for restoring the contact cache
	ContactAllocator contact_allocator(GetContactAllocator());

	// When validating, get all existing body pairs
	Array<const BPKeyValue *> all_bp;
	if (inStream.IsValidating())
		inReadCache.GetAllBodyPairsSorted(all_bp);

	// Read amount of body pairs
	uint32 num_body_pairs;
	if (inStream.IsValidating())
		num_body_pairs = uint32(all_bp.size());
	inStream.Read(num_body_pairs);

	// Read entire cache
	for (uint32 i = 0; i < num_body_pairs; ++i)
	{
		// Read key
		BodyPair body_pair_key;
		if (inStream.IsValidating() && i < all_bp.size())
			body_pair_key = all_bp[i]->GetKey();
		inStream.Read(body_pair_key);

		// Check if we want to restore this contact
		if (inFilter == nullptr || inFilter->ShouldRestoreContact(body_pair_key.mBodyA, body_pair_key.mBodyB))
		{
			// Create new entry for this body pair
			uint64 body_pair_hash = body_pair_key.GetHash();
			BPKeyValue *bp_kv = Create(contact_allocator, body_pair_key, body_pair_hash);
			if (bp_kv == nullptr)
			{
				// Out of cache space
				success = false;
				break;
			}
			CachedBodyPair &bp = bp_kv->GetValue();

			// Read body pair
			if (inStream.IsValidating() && i < all_bp.size())
				memcpy(&bp, &all_bp[i]->GetValue(), sizeof(CachedBodyPair));
			bp.RestoreState(inStream);

			// When validating, get all existing manifolds
			Array<const MKeyValue *> all_m;
			if (inStream.IsValidating())
				inReadCache.GetAllManifoldsSorted(all_bp[i]->GetValue(), all_m);

			// Read amount of manifolds
			uint32 num_manifolds = 0;
			if (inStream.IsValidating())
				num_manifolds = uint32(all_m.size());
			inStream.Read(num_manifolds);

			uint32 handle = ManifoldMap::cInvalidHandle;
			for (uint32 j = 0; j < num_manifolds; ++j)
			{
				// Read key
				SubShapeIDPair sub_shape_key;
				if (inStream.IsValidating() && j < all_m.size())
					sub_shape_key = all_m[j]->GetKey();
				inStream.Read(sub_shape_key);
				uint64 sub_shape_key_hash = sub_shape_key.GetHash();

				// Read amount of contact points
				uint16 num_contact_points = 0;
				if (inStream.IsValidating() && j < all_m.size())
					num_contact_points = all_m[j]->GetValue().mNumContactPoints;
				inStream.Read(num_contact_points);

				// CRUMB: whether a soft-contact tail follows the points
				bool soft_tail = false;
				if (inStream.IsValidating() && j < all_m.size())
					soft_tail = all_m[j]->GetValue().HasSoftTail();
				inStream.Read(soft_tail);

				// Read manifold
				MKeyValue *m_kv = Create(contact_allocator, sub_shape_key, sub_shape_key_hash, num_contact_points, soft_tail);
				if (m_kv == nullptr)
				{
					// Out of cache space
					success = false;
					break;
				}
				CachedManifold &cm = m_kv->GetValue();
				if (inStream.IsValidating() && j < all_m.size())
				{
					memcpy(&cm, &all_m[j]->GetValue(), CachedManifold::sGetRequiredTotalSize(num_contact_points) + (soft_tail? sizeof(SoftContactTail) : 0));
					cm.mNumContactPoints = uint16(num_contact_points); // Restore num contact points
				}
				cm.RestoreState(inStream);
				cm.mNextWithSameBodyPair = handle;
				handle = ToHandle(m_kv);

				// Read contact points
				for (uint32 k = 0; k < num_contact_points; ++k)
					cm.mContactPoints[k].RestoreState(inStream);

				// CRUMB: and the tail
				if (soft_tail)
					inStream.Read(*cm.GetSoftTail());
			}
			bp.mFirstCachedManifold = handle;
		}
		else
		{
			// Skip the contact
			CachedBodyPair bp;
			bp.RestoreState(inStream);
			uint32 num_manifolds = 0;
			inStream.Read(num_manifolds);
			for (uint32 j = 0; j < num_manifolds; ++j)
			{
				SubShapeIDPair sub_shape_key;
				inStream.Read(sub_shape_key);
				uint16 num_contact_points;
				inStream.Read(num_contact_points);
				bool soft_tail = false; // CRUMB
				inStream.Read(soft_tail);
				CachedManifold cm;
				cm.RestoreState(inStream);
				for (uint32 k = 0; k < num_contact_points; ++k)
					cm.mContactPoints[0].RestoreState(inStream);
				if (soft_tail)
				{
					SoftContactTail tail;
					inStream.Read(tail);
				}
			}
		}
	}

	// When validating, get all existing CCD manifolds
	Array<const MKeyValue *> all_m;
	if (inStream.IsValidating())
		inReadCache.GetAllCCDManifoldsSorted(all_m);

	// Read amount of CCD manifolds
	uint32 num_manifolds;
	if (inStream.IsValidating())
		num_manifolds = uint32(all_m.size());
	inStream.Read(num_manifolds);

	for (uint32 j = 0; j < num_manifolds; ++j)
	{
		// Read key
		SubShapeIDPair sub_shape_key;
		if (inStream.IsValidating() && j < all_m.size())
			sub_shape_key = all_m[j]->GetKey();
		inStream.Read(sub_shape_key);

		// Check if we want to restore this contact
		if (inFilter == nullptr || inFilter->ShouldRestoreContact(sub_shape_key.GetBody1ID(), sub_shape_key.GetBody2ID()))
		{
			// Create CCD manifold
			uint64 sub_shape_key_hash = sub_shape_key.GetHash();
			MKeyValue *m_kv = Create(contact_allocator, sub_shape_key, sub_shape_key_hash, 0);
			if (m_kv == nullptr)
			{
				// Out of cache space
				success = false;
				break;
			}
			CachedManifold &cm = m_kv->GetValue();
			cm.mFlags |= (uint16)CachedManifold::EFlags::CCDContact;
		}
	}

#ifdef JPH_ENABLE_ASSERTS
	// We don't finalize until the last part is restored
	if (inStream.IsLastPart())
		mIsFinalized = true;
#endif

	return success;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager
////////////////////////////////////////////////////////////////////////////////////////////////////////

ContactConstraintManager::ContactConstraintManager(const PhysicsSettings &inPhysicsSettings) :
	mPhysicsSettings(inPhysicsSettings)
{
#ifdef JPH_ENABLE_ASSERTS
	// For the first frame mark this empty buffer as finalized
	mCache[mCacheWriteIdx ^ 1].Finalize();
#endif
}

ContactConstraintManager::~ContactConstraintManager()
{
	JPH_ASSERT(mConstraints == nullptr);
}

void ContactConstraintManager::Init(uint inMaxBodyPairs, uint inMaxContactConstraints)
{
	// Limit the number of constraints so that the allocation size fits in an unsigned integer
	mMaxConstraints = min(inMaxContactConstraints, cMaxContactConstraintsLimit);
	JPH_ASSERT(mMaxConstraints == inMaxContactConstraints, "Cannot support this many contact constraints!");

	// Calculate worst case cache usage (CRUMB: with a soft-contact tail)
	constexpr uint cMaxManifoldSizePerConstraint = sizeof(CachedManifold) + (MaxContactPoints - 1) * sizeof(CachedContactPoint) + sizeof(SoftContactTail);
	static_assert(cMaxManifoldSizePerConstraint < sizeof(ContactConstraint)); // If not true, then the next line can overflow
	uint cached_manifolds_size = mMaxConstraints * cMaxManifoldSizePerConstraint;

	// Init the caches
	mCache[0].Init(inMaxBodyPairs, mMaxConstraints, cached_manifolds_size);
	mCache[1].Init(inMaxBodyPairs, mMaxConstraints, cached_manifolds_size);
}

void ContactConstraintManager::PrepareConstraintBuffer(PhysicsUpdateContext *inContext)
{
	// Store context
	mUpdateContext = inContext;

	// Allocate temporary constraint buffer
	JPH_ASSERT(mConstraints == nullptr);
	mConstraints = (ContactConstraint *)inContext->mTempAllocator->Allocate(mMaxConstraints * sizeof(ContactConstraint));

	// CRUMB: and the soft patches' side table, freed before the constraints (FinishConstraintBuffer)
	JPH_ASSERT(mSoftPatches == nullptr);
	if (mMaxSoftPatches > 0)
		mSoftPatches = (SoftPatchConstraint *)inContext->mTempAllocator->Allocate(mMaxSoftPatches * sizeof(SoftPatchConstraint));
	mNumSoftPatches = 0;
	mSoftStats = SoftContactStats();
}

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE void ContactConstraintManager::TemplatedCalculateFrictionAndNonPenetrationConstraintProperties(ContactConstraint &ioConstraint, const ContactSettings &inSettings, float inDeltaTime, Vec3Arg inGravity, RMat44Arg inTransformBody1, RMat44Arg inTransformBody2, const Body &inBody1, const Body &inBody2)
{
	// Calculate scaled mass and inertia
	Mat44 inv_i1;
	if constexpr (Type1 == EMotionType::Dynamic)
	{
		const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
		inv_i1 = inSettings.mInvInertiaScale1 * mp1->GetInverseInertiaForRotation(inTransformBody1.GetRotation());
	}
	else
	{
		inv_i1 = Mat44::sZero();
	}

	Mat44 inv_i2;
	if constexpr (Type2 == EMotionType::Dynamic)
	{
		const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();
		inv_i2 = inSettings.mInvInertiaScale2 * mp2->GetInverseInertiaForRotation(inTransformBody2.GetRotation());
	}
	else
	{
		inv_i2 = Mat44::sZero();
	}

	// Calculate tangents
	Vec3 t1, t2;
	ioConstraint.GetTangents(t1, t2);

	Vec3 ws_normal = ioConstraint.GetWorldSpaceNormal();

	// Setup velocity constraint properties
	float min_velocity_for_restitution = mPhysicsSettings.mMinVelocityForRestitution;
	for (WorldContactPoint &wcp : ioConstraint.mContactPoints)
	{
		RVec3 p1 = inTransformBody1 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition1);
		RVec3 p2 = inTransformBody2 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition2);
		wcp.TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<Type1, Type2>(inDeltaTime, inGravity, inBody1, inBody2, ioConstraint.mInvMass1, ioConstraint.mInvMass2, inv_i1, inv_i2, p1, p2, ws_normal, t1, t2, inSettings, min_velocity_for_restitution);
	}
}

inline void ContactConstraintManager::CalculateFrictionAndNonPenetrationConstraintProperties(ContactConstraint &ioConstraint, const ContactSettings &inSettings, float inDeltaTime, Vec3Arg inGravity, RMat44Arg inTransformBody1, RMat44Arg inTransformBody2, const Body &inBody1, const Body &inBody2)
{
	// Dispatch to the correct templated form
	switch (inBody1.GetMotionType())
	{
	case EMotionType::Dynamic:
		switch (inBody2.GetMotionType())
		{
		case EMotionType::Dynamic:
			TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<EMotionType::Dynamic, EMotionType::Dynamic>(ioConstraint, inSettings, inDeltaTime, inGravity, inTransformBody1, inTransformBody2, inBody1, inBody2);
			break;

		case EMotionType::Kinematic:
			TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<EMotionType::Dynamic, EMotionType::Kinematic>(ioConstraint, inSettings, inDeltaTime, inGravity, inTransformBody1, inTransformBody2, inBody1, inBody2);
			break;

		case EMotionType::Static:
			TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<EMotionType::Dynamic, EMotionType::Static>(ioConstraint, inSettings, inDeltaTime, inGravity, inTransformBody1, inTransformBody2, inBody1, inBody2);
			break;

		default:
			JPH_ASSERT(false);
			break;
		}
		break;

	case EMotionType::Kinematic:
		JPH_ASSERT(inBody2.IsDynamic());
		TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<EMotionType::Kinematic, EMotionType::Dynamic>(ioConstraint, inSettings, inDeltaTime, inGravity, inTransformBody1, inTransformBody2, inBody1, inBody2);
		break;

	case EMotionType::Static:
		JPH_ASSERT(inBody2.IsDynamic());
		TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<EMotionType::Static, EMotionType::Dynamic>(ioConstraint, inSettings, inDeltaTime, inGravity, inTransformBody1, inTransformBody2, inBody1, inBody2);
		break;

	default:
		JPH_ASSERT(false);
		break;
	}
}

void ContactConstraintManager::GetContactsFromCache(ContactAllocator &ioContactAllocator, Body &inBody1, Body &inBody2, bool &outPairHandled, bool &outConstraintCreated)
{
	// Start with nothing found and not handled
	outConstraintCreated = false;
	outPairHandled = false;

	// CRUMB: a body with compliant shapes always has its contacts built from the current poses: its patch is
	// the law at this step's squash, so a cached manifold would replay a stale one
	if (inBody1.HasCompliantShapes() || inBody2.HasCompliantShapes())
		return;

	// Swap bodies so that body 1 id < body 2 id
	Body *body1, *body2;
	if (inBody1.GetID() < inBody2.GetID())
	{
		body1 = &inBody1;
		body2 = &inBody2;
	}
	else
	{
		body1 = &inBody2;
		body2 = &inBody1;
	}

	// Find the cached body pair
	BodyPair body_pair_key(body1->GetID(), body2->GetID());
	uint64 body_pair_hash = body_pair_key.GetHash();
	const ManifoldCache &read_cache = mCache[mCacheWriteIdx ^ 1];
	const BPKeyValue *kv = read_cache.Find(body_pair_key, body_pair_hash);
	if (kv == nullptr)
		return;
	const CachedBodyPair &input_cbp = kv->GetValue();

	// Get relative translation
	Quat inv_r1 = body1->GetRotation().Conjugated();
	Vec3 delta_position = inv_r1 * Vec3(body2->GetCenterOfMassPosition() - body1->GetCenterOfMassPosition());

	// Get old position delta
	Vec3 old_delta_position = Vec3::sLoadFloat3Unsafe(input_cbp.mDeltaPosition);

	// Check if bodies are still roughly in the same relative position
	if ((delta_position - old_delta_position).LengthSq() > mPhysicsSettings.mBodyPairCacheMaxDeltaPositionSq)
		return;

	// Determine relative orientation
	Quat delta_rotation = inv_r1 * body2->GetRotation();

	// Reconstruct old quaternion delta
	Quat old_delta_rotation = Quat::sLoadFloat3Unsafe(input_cbp.mDeltaRotation);

	// Check if bodies are still roughly in the same relative orientation
	// The delta between 2 quaternions p and q is: p q^* = [rotation_axis * sin(angle / 2), cos(angle / 2)]
	// From the W component we can extract the angle: cos(angle / 2) = px * qx + py * qy + pz * qz + pw * qw = p . q
	// Since we want to abort if the rotation is smaller than -angle or bigger than angle, we can write the comparison as |p . q| < cos(angle / 2)
	if (abs(delta_rotation.Dot(old_delta_rotation)) < mPhysicsSettings.mBodyPairCacheCosMaxDeltaRotationDiv2)
		return;

	// The cache is valid, return that we've handled this body pair
	outPairHandled = true;

	// Copy the cached body pair to this frame
	ManifoldCache &write_cache = mCache[mCacheWriteIdx];
	BPKeyValue *output_bp_kv = write_cache.Create(ioContactAllocator, body_pair_key, body_pair_hash);
	if (output_bp_kv == nullptr)
		return; // Out of cache space
	CachedBodyPair *output_cbp = &output_bp_kv->GetValue();
	memcpy(output_cbp, &input_cbp, sizeof(CachedBodyPair));

	// If there were no contacts, we have handled the contact
	if (input_cbp.mFirstCachedManifold == ManifoldMap::cInvalidHandle)
		return;

	// Get body transforms
	RMat44 transform_body1 = body1->GetCenterOfMassTransform();
	RMat44 transform_body2 = body2->GetCenterOfMassTransform();

	// Get time step and gravity
	float delta_time = mUpdateContext->mStepDeltaTime;
	Vec3 gravity = mUpdateContext->mPhysicsSystem->GetGravity();

	// Copy manifolds
	uint32 output_handle = ManifoldMap::cInvalidHandle;
	uint32 input_handle = input_cbp.mFirstCachedManifold;
	do
	{
		JPH_PROFILE("Add Constraint From Cached Manifold");

		// Find the existing manifold
		const MKeyValue *input_kv = read_cache.FromHandle(input_handle);
		const SubShapeIDPair &input_key = input_kv->GetKey();
		const CachedManifold &input_cm = input_kv->GetValue();
		JPH_ASSERT(input_cm.mNumContactPoints > 0); // There should be contact points in this manifold!

		// Create room for manifold in write buffer and copy data
		uint64 input_hash = input_key.GetHash();
		MKeyValue *output_kv = write_cache.Create(ioContactAllocator, input_key, input_hash, input_cm.mNumContactPoints);
		if (output_kv == nullptr)
			break; // Out of cache space
		CachedManifold *output_cm = &output_kv->GetValue();
		memcpy(output_cm, &input_cm, CachedManifold::sGetRequiredTotalSize(input_cm.mNumContactPoints));

		// Link the object under the body pairs
		output_cm->mNextWithSameBodyPair = output_handle;
		output_handle = write_cache.ToHandle(output_kv);

		// Calculate default contact settings
		ContactSettings settings;
		settings.mCombinedFriction = mCombineFriction(*body1, input_key.GetSubShapeID1(), *body2, input_key.GetSubShapeID2());
		settings.mCombinedRestitution = mCombineRestitution(*body1, input_key.GetSubShapeID1(), *body2, input_key.GetSubShapeID2());
		settings.mIsSensor = body1->IsSensor() || body2->IsSensor();

		// Calculate world space contact normal
		Vec3 world_space_normal = transform_body2.Multiply3x3(Vec3::sLoadFloat3Unsafe(output_cm->mContactNormal)).Normalized();

		// Call contact listener to update settings
		if (mContactListener != nullptr)
		{
			// Convert constraint to manifold structure for callback
			ContactManifold manifold;
			manifold.mWorldSpaceNormal = world_space_normal;
			manifold.mSubShapeID1 = input_key.GetSubShapeID1();
			manifold.mSubShapeID2 = input_key.GetSubShapeID2();
			manifold.mBaseOffset = transform_body1.GetTranslation();
			manifold.mRelativeContactPointsOn1.resize(output_cm->mNumContactPoints);
			manifold.mRelativeContactPointsOn2.resize(output_cm->mNumContactPoints);
			Mat44 local_transform_body2 = transform_body2.PostTranslated(-manifold.mBaseOffset).ToMat44();
			float penetration_depth = -FLT_MAX;
			for (uint32 i = 0; i < output_cm->mNumContactPoints; ++i)
			{
				const CachedContactPoint &ccp = output_cm->mContactPoints[i];
				manifold.mRelativeContactPointsOn1[i] = transform_body1.Multiply3x3(Vec3::sLoadFloat3Unsafe(ccp.mPosition1));
				manifold.mRelativeContactPointsOn2[i] = local_transform_body2 * Vec3::sLoadFloat3Unsafe(ccp.mPosition2);
				penetration_depth = max(penetration_depth, (manifold.mRelativeContactPointsOn1[i] - manifold.mRelativeContactPointsOn2[i]).Dot(world_space_normal));
			}
			manifold.mPenetrationDepth = penetration_depth; // We don't have the penetration depth anymore, estimate it

			// Notify callback
			mContactListener->OnContactPersisted(*body1, *body2, manifold, settings);
		}

		JPH_ASSERT(settings.mIsSensor || !(body1->IsSensor() || body2->IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
		if (!settings.mIsSensor // If one of the bodies is a sensor, don't actually create the constraint
			&& ((body1->IsDynamic() && settings.mInvMassScale1 != 0.0f) // One of the bodies must have mass to be able to create a contact constraint
				|| (body2->IsDynamic() && settings.mInvMassScale2 != 0.0f)))
		{
			// Add contact constraint in world space for the solver
			uint32 constraint_idx = mNumConstraints++;
			if (constraint_idx >= mMaxConstraints)
			{
				ioContactAllocator.mErrors |= EPhysicsUpdateError::ContactConstraintsFull;
				break;
			}

			// A constraint will be created
			outConstraintCreated = true;

			ContactConstraint &constraint = mConstraints[constraint_idx];
			new (&constraint) ContactConstraint();
			constraint.mBody1 = body1;
			constraint.mBody2 = body2;
			constraint.mSortKey = input_hash;
			world_space_normal.StoreFloat3(&constraint.mWorldSpaceNormal);
			constraint.mCombinedFriction = settings.mCombinedFriction;
			constraint.mInvMass1 = body1->GetMotionPropertiesUnchecked() != nullptr? settings.mInvMassScale1 * body1->GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
			constraint.mInvInertiaScale1 = settings.mInvInertiaScale1;
			constraint.mInvMass2 = body2->GetMotionPropertiesUnchecked() != nullptr? settings.mInvMassScale2 * body2->GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
			constraint.mInvInertiaScale2 = settings.mInvInertiaScale2;
			constraint.mSubShapeID1 = input_key.GetSubShapeID1();
			constraint.mSubShapeID2 = input_key.GetSubShapeID2();
			constraint.mReportSolved = settings.mReportSolved;
			constraint.mContactPoints.resize(output_cm->mNumContactPoints);
			for (uint32 i = 0; i < output_cm->mNumContactPoints; ++i)
			{
				CachedContactPoint &ccp = output_cm->mContactPoints[i];
				WorldContactPoint &wcp = constraint.mContactPoints[i];
				wcp.mNonPenetrationConstraint.SetTotalLambda(ccp.mNonPenetrationLambda);
				wcp.mFrictionConstraint1.SetTotalLambda(ccp.mFrictionLambda[0]);
				wcp.mFrictionConstraint2.SetTotalLambda(ccp.mFrictionLambda[1]);
				wcp.mContactPoint = &ccp;
			}

			JPH_DET_LOG("GetContactsFromCache: id1: " << constraint.mBody1->GetID() << " id2: " << constraint.mBody2->GetID() << " key: " << constraint.mSortKey);

			// Calculate friction and non-penetration constraint properties for all contact points
			CalculateFrictionAndNonPenetrationConstraintProperties(constraint, settings, delta_time, gravity, transform_body1, transform_body2, *body1, *body2);

			// Notify island builder
			mUpdateContext->mIslandBuilder->LinkContact(constraint_idx, body1->GetIndexInActiveBodiesInternal(), body2->GetIndexInActiveBodiesInternal());

		#ifdef JPH_DEBUG_RENDERER
			// Draw the manifold
			if (sDrawContactManifolds)
				constraint.Draw(DebugRenderer::sInstance, Color::sYellow);
		#endif // JPH_DEBUG_RENDERER
		}

		// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
		input_cm.mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;

		// Fetch the next manifold
		input_handle = input_cm.mNextWithSameBodyPair;
	}
	while (input_handle != ManifoldMap::cInvalidHandle);
	output_cbp->mFirstCachedManifold = output_handle;
}

ContactConstraintManager::BodyPairHandle ContactConstraintManager::AddBodyPair(ContactAllocator &ioContactAllocator, const Body &inBody1, const Body &inBody2)
{
	// Swap bodies so that body 1 id < body 2 id
	const Body *body1, *body2;
	if (inBody1.GetID() < inBody2.GetID())
	{
		body1 = &inBody1;
		body2 = &inBody2;
	}
	else
	{
		body1 = &inBody2;
		body2 = &inBody1;
	}

	// Add an entry
	BodyPair body_pair_key(body1->GetID(), body2->GetID());
	uint64 body_pair_hash = body_pair_key.GetHash();
	BPKeyValue *body_pair_kv = mCache[mCacheWriteIdx].Create(ioContactAllocator, body_pair_key, body_pair_hash);
	if (body_pair_kv == nullptr)
		return nullptr; // Out of cache space
	CachedBodyPair *cbp = &body_pair_kv->GetValue();
	cbp->mFirstCachedManifold = ManifoldMap::cInvalidHandle;

	// Get relative translation
	Quat inv_r1 = body1->GetRotation().Conjugated();
	Vec3 delta_position = inv_r1 * Vec3(body2->GetCenterOfMassPosition() - body1->GetCenterOfMassPosition());

	// Store it
	delta_position.StoreFloat3(&cbp->mDeltaPosition);

	// Determine relative orientation
	Quat delta_rotation = inv_r1 * body2->GetRotation();

	// Store it
	delta_rotation.StoreFloat3(&cbp->mDeltaRotation);

	return cbp;
}

template <EMotionType Type1, EMotionType Type2>
bool ContactConstraintManager::TemplatedAddContactConstraint(ContactAllocator &ioContactAllocator, BodyPairHandle inBodyPairHandle, Body &inBody1, Body &inBody2, const ContactManifold &inManifold)
{
	// Calculate hash
	SubShapeIDPair key { inBody1.GetID(), inManifold.mSubShapeID1, inBody2.GetID(), inManifold.mSubShapeID2 };
	uint64 key_hash = key.GetHash();

	// Determine number of contact points
	int num_contact_points = (int)inManifold.mRelativeContactPointsOn1.size();
	JPH_ASSERT(num_contact_points <= MaxContactPoints);
	JPH_ASSERT(num_contact_points == (int)inManifold.mRelativeContactPointsOn2.size());

	// CRUMB: a manifold of a body with compliant shapes reserves a soft-contact tail after its points
	bool soft_tail = inBody1.HasCompliantShapes() || inBody2.HasCompliantShapes();

	// Reserve space for new contact cache entry
	// Note that for dynamic vs dynamic we always require the first body to have a lower body id to get a consistent key
	// under which to look up the contact
	ManifoldCache &write_cache = mCache[mCacheWriteIdx];
	MKeyValue *new_manifold_kv = write_cache.Create(ioContactAllocator, key, key_hash, num_contact_points, soft_tail);
	if (new_manifold_kv == nullptr)
		return false; // Out of cache space
	CachedManifold *new_manifold = &new_manifold_kv->GetValue();

	// Transform the world space normal to the space of body 2 (this is usually the static body)
	RMat44 inverse_transform_body2 = inBody2.GetInverseCenterOfMassTransform();
	inverse_transform_body2.Multiply3x3(inManifold.mWorldSpaceNormal).Normalized().StoreFloat3(&new_manifold->mContactNormal);

	// Settings object that gets passed to the callback
	ContactSettings settings;
	settings.mCombinedFriction = mCombineFriction(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	settings.mCombinedRestitution = mCombineRestitution(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	settings.mIsSensor = inBody1.IsSensor() || inBody2.IsSensor();

	// CRUMB: the record the listener may fill with a soft patch
	SoftContactPatch soft_patch;
	if (soft_tail)
		settings.mSoftPatch = &soft_patch;

	// Get the contact points for the old cache entry
	const ManifoldCache &read_cache = mCache[mCacheWriteIdx ^ 1];
	const MKeyValue *old_manifold_kv = read_cache.Find(key, key_hash);
	const CachedContactPoint *ccp_start;
	const CachedContactPoint *ccp_end;
	const SoftContactTail *old_tail = nullptr; // CRUMB
	if (old_manifold_kv != nullptr)
	{
		// CRUMB: last step's tail, when this manifold carried one
		const CachedManifold *old_manifold = &old_manifold_kv->GetValue();
		if (soft_tail && old_manifold->HasSoftTail() && old_manifold->mNumContactPoints > 0)
			old_tail = old_manifold->GetSoftTail();
		settings.mSoftTail = old_tail;

		// Call point persisted listener
		if (mContactListener != nullptr)
			mContactListener->OnContactPersisted(inBody1, inBody2, inManifold, settings);

		// Fetch the contact points from the old manifold
		ccp_start = old_manifold->mContactPoints;
		ccp_end = ccp_start + old_manifold->mNumContactPoints;

		// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
		old_manifold->mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;
	}
	else
	{
		// Call point added listener
		if (mContactListener != nullptr)
			mContactListener->OnContactAdded(inBody1, inBody2, inManifold, settings);

		// No contact points available from old manifold
		ccp_start = nullptr;
		ccp_end = nullptr;
	}

	// CRUMB: this step's tail starts as last step's; a new episode starts with its first touch undecided
	SoftContactTail *new_tail = nullptr;
	if (soft_tail)
	{
		new_tail = new_manifold->GetSoftTail();
		if (old_tail != nullptr)
			memcpy(new_tail, old_tail, sizeof(SoftContactTail));
		else
		{
			memset(new_tail, 0, sizeof(SoftContactTail));
			new_tail->mFlags = SoftContactTail::OnsetPending;
		}
		new_tail->mFlags &= ~uint32(SoftContactTail::Solved);
	}

	// Get inverse transform for body 1
	RMat44 inverse_transform_body1 = inBody1.GetInverseCenterOfMassTransform();

	bool contact_constraint_created = false;

	// If one of the bodies is a sensor, don't actually create the constraint
	JPH_ASSERT(settings.mIsSensor || !(inBody1.IsSensor() || inBody2.IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
	if (!settings.mIsSensor
		&& ((inBody1.IsDynamic() && settings.mInvMassScale1 != 0.0f) // One of the bodies must have mass to be able to create a contact constraint
			|| (inBody2.IsDynamic() && settings.mInvMassScale2 != 0.0f)))
	{
		// Add contact constraint
		uint32 constraint_idx = mNumConstraints++;
		if (constraint_idx >= mMaxConstraints)
		{
			ioContactAllocator.mErrors |= EPhysicsUpdateError::ContactConstraintsFull;

			// Manifold has been created already, we're not filling it in, so we need to reset the contact number of points.
			// Note that we don't hook it up to the body pair cache so that it won't be used as a cache during the next simulation.
			new_manifold->mNumContactPoints = 0;
			new_manifold->mFlags.fetch_and(uint16(~uint16(CachedManifold::EFlags::SoftTail)), memory_order_relaxed); // CRUMB: the tail moved with the point count
			return false;
		}

		// We will create a contact constraint
		contact_constraint_created = true;

		ContactConstraint &constraint = mConstraints[constraint_idx];
		new (&constraint) ContactConstraint();
		constraint.mBody1 = &inBody1;
		constraint.mBody2 = &inBody2;
		constraint.mSortKey = key_hash;
		inManifold.mWorldSpaceNormal.StoreFloat3(&constraint.mWorldSpaceNormal);
		constraint.mCombinedFriction = settings.mCombinedFriction;
		constraint.mInvMass1 = inBody1.GetMotionPropertiesUnchecked() != nullptr? settings.mInvMassScale1 * inBody1.GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
		constraint.mInvInertiaScale1 = settings.mInvInertiaScale1;
		constraint.mInvMass2 = inBody2.GetMotionPropertiesUnchecked() != nullptr? settings.mInvMassScale2 * inBody2.GetMotionPropertiesUnchecked()->GetInverseMassUnchecked() : 0.0f;
		constraint.mInvInertiaScale2 = settings.mInvInertiaScale2;
		constraint.mSubShapeID1 = inManifold.mSubShapeID1;
		constraint.mSubShapeID2 = inManifold.mSubShapeID2;
		constraint.mReportSolved = settings.mReportSolved;

		JPH_DET_LOG("TemplatedAddContactConstraint: id1: " << constraint.mBody1->GetID() << " id2: " << constraint.mBody2->GetID() << " key: " << constraint.mSortKey);

		// Notify island builder
		mUpdateContext->mIslandBuilder->LinkContact(constraint_idx, inBody1.GetIndexInActiveBodiesInternal(), inBody2.GetIndexInActiveBodiesInternal());

		// Get time step and gravity
		float delta_time = mUpdateContext->mStepDeltaTime;
		Vec3 gravity = mUpdateContext->mPhysicsSystem->GetGravity();

		// Calculate scaled mass and inertia
		float inv_m1;
		Mat44 inv_i1;
		if constexpr (Type1 == EMotionType::Dynamic)
		{
			const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
			inv_m1 = settings.mInvMassScale1 * mp1->GetInverseMass();
			inv_i1 = settings.mInvInertiaScale1 * mp1->GetInverseInertiaForRotation(inverse_transform_body1.Transposed3x3());
		}
		else
		{
			inv_m1 = 0.0f;
			inv_i1 = Mat44::sZero();
		}

		float inv_m2;
		Mat44 inv_i2;
		if constexpr (Type2 == EMotionType::Dynamic)
		{
			const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();
			inv_m2 = settings.mInvMassScale2 * mp2->GetInverseMass();
			inv_i2 = settings.mInvInertiaScale2 * mp2->GetInverseInertiaForRotation(inverse_transform_body2.Transposed3x3());
		}
		else
		{
			inv_m2 = 0.0f;
			inv_i2 = Mat44::sZero();
		}

		// CRUMB: a soft patch replaces the rows of the manifold's points. The points are still cached, with no
		// impulses: they serve the removal callbacks and nothing else.
		bool soft_solved = false;
		if (soft_tail && soft_patch.mEnabled)
		{
			uint32 soft_idx = mSoftPatches != nullptr? mNumSoftPatches++ : mMaxSoftPatches;
			if (soft_idx < mMaxSoftPatches)
			{
				constraint.mSoftPatchIndex = soft_idx;
				bool warm = old_tail != nullptr && (old_tail->mFlags & SoftContactTail::Solved) != 0;
				SetupSoftPatch(constraint, mSoftPatches[soft_idx], soft_patch, *new_tail, warm, inv_m1, inv_i1, inv_m2, inv_i2, delta_time, gravity);
				for (int i = 0; i < num_contact_points; ++i)
				{
					CachedContactPoint &cp = new_manifold->mContactPoints[i];
					Vec3(inverse_transform_body1 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn1[i])).StoreFloat3(&cp.mPosition1);
					Vec3(inverse_transform_body2 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn2[i])).StoreFloat3(&cp.mPosition2);
					cp.mNonPenetrationLambda = 0.0f;
					cp.mFrictionLambda[0] = 0.0f;
					cp.mFrictionLambda[1] = 0.0f;
				}
				soft_solved = true;
			}
			else
				++mSoftStats.mOverflow;
		}

		// Calculate tangents
		Vec3 t1, t2;
		constraint.GetTangents(t1, t2);

		constraint.mContactPoints.resize(soft_solved? 0 : num_contact_points);
		for (int i = 0; i < (soft_solved? 0 : num_contact_points); ++i)
		{
			// Convert to world space and set positions
			WorldContactPoint &wcp = constraint.mContactPoints[i];
			RVec3 p1_ws = inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn1[i];
			RVec3 p2_ws = inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn2[i];

			// Convert to local space to the body
			Vec3 p1_ls = Vec3(inverse_transform_body1 * p1_ws);
			Vec3 p2_ls = Vec3(inverse_transform_body2 * p2_ws);

			// Check if we have a close contact point from last update
			bool lambda_set = false;
			for (const CachedContactPoint *ccp = ccp_start; ccp < ccp_end; ccp++)
				if (Vec3::sLoadFloat3Unsafe(ccp->mPosition1).IsClose(p1_ls, mPhysicsSettings.mContactPointPreserveLambdaMaxDistSq)
					&& Vec3::sLoadFloat3Unsafe(ccp->mPosition2).IsClose(p2_ls, mPhysicsSettings.mContactPointPreserveLambdaMaxDistSq))
				{
					// Get lambdas from previous frame
					wcp.mNonPenetrationConstraint.SetTotalLambda(ccp->mNonPenetrationLambda);
					wcp.mFrictionConstraint1.SetTotalLambda(ccp->mFrictionLambda[0]);
					wcp.mFrictionConstraint2.SetTotalLambda(ccp->mFrictionLambda[1]);
					lambda_set = true;
					break;
				}
			if (!lambda_set)
			{
				wcp.mNonPenetrationConstraint.SetTotalLambda(0.0f);
				wcp.mFrictionConstraint1.SetTotalLambda(0.0f);
				wcp.mFrictionConstraint2.SetTotalLambda(0.0f);
			}

			// Create new contact point
			CachedContactPoint &cp = new_manifold->mContactPoints[i];
			p1_ls.StoreFloat3(&cp.mPosition1);
			p2_ls.StoreFloat3(&cp.mPosition2);
			wcp.mContactPoint = &cp;

			// Setup velocity constraint
			wcp.TemplatedCalculateFrictionAndNonPenetrationConstraintProperties<Type1, Type2>(delta_time, gravity, inBody1, inBody2, inv_m1, inv_m2, inv_i1, inv_i2, p1_ws, p2_ws, inManifold.mWorldSpaceNormal, t1, t2, settings, mPhysicsSettings.mMinVelocityForRestitution);
		}

	#ifdef JPH_DEBUG_RENDERER
		// Draw the manifold
		if (sDrawContactManifolds)
			constraint.Draw(DebugRenderer::sInstance, Color::sOrange);
	#endif // JPH_DEBUG_RENDERER
	}
	else
	{
		// Store the contact manifold in the cache
		for (int i = 0; i < num_contact_points; ++i)
		{
			// Convert to local space to the body
			Vec3 p1 = Vec3(inverse_transform_body1 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn1[i]));
			Vec3 p2 = Vec3(inverse_transform_body2 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn2[i]));

			// Create new contact point
			CachedContactPoint &cp = new_manifold->mContactPoints[i];
			p1.StoreFloat3(&cp.mPosition1);
			p2.StoreFloat3(&cp.mPosition2);

			// Reset contact impulses, we haven't applied any
			cp.mNonPenetrationLambda = 0.0f;
			cp.mFrictionLambda[0] = 0.0f;
			cp.mFrictionLambda[1] = 0.0f;
		}
	}

	// Store cached contact point in body pair cache
	CachedBodyPair *cbp = reinterpret_cast<CachedBodyPair *>(inBodyPairHandle);
	new_manifold->mNextWithSameBodyPair = cbp->mFirstCachedManifold;
	cbp->mFirstCachedManifold = write_cache.ToHandle(new_manifold_kv);

	// A contact constraint was added
	return contact_constraint_created;
}

bool ContactConstraintManager::AddContactConstraint(ContactAllocator &ioContactAllocator, BodyPairHandle inBodyPairHandle, Body &inBody1, Body &inBody2, const ContactManifold &inManifold)
{
	JPH_PROFILE_FUNCTION();

	JPH_DET_LOG("AddContactConstraint: id1: " << inBody1.GetID() << " id2: " << inBody2.GetID()
		<< " subshape1: " << inManifold.mSubShapeID1 << " subshape2: " << inManifold.mSubShapeID2
		<< " normal: " << inManifold.mWorldSpaceNormal << " pendepth: " << inManifold.mPenetrationDepth);

	JPH_ASSERT(inManifold.mWorldSpaceNormal.IsNormalized());

	// Swap bodies so that body 1 id < body 2 id
	const ContactManifold *manifold;
	Body *body1, *body2;
	ContactManifold temp;
	if (inBody2.GetID() < inBody1.GetID())
	{
		body1 = &inBody2;
		body2 = &inBody1;
		temp = inManifold.SwapShapes();
		manifold = &temp;
	}
	else
	{
		body1 = &inBody1;
		body2 = &inBody2;
		manifold = &inManifold;
	}

	// Dispatch to the correct templated form
	// Note: Non-dynamic vs non-dynamic can happen in this case due to one body being a sensor, so we need to have an extended switch case here
	switch (body1->GetMotionType())
	{
	case EMotionType::Dynamic:
		{
			switch (body2->GetMotionType())
			{
			case EMotionType::Dynamic:
				return TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Dynamic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

			case EMotionType::Kinematic:
				return TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Kinematic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

			case EMotionType::Static:
				return TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Static>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

			default:
				JPH_ASSERT(false);
				break;
			}
			break;
		}

	case EMotionType::Kinematic:
		switch (body2->GetMotionType())
		{
		case EMotionType::Dynamic:
			return TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Dynamic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

		case EMotionType::Kinematic:
			return TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Kinematic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

		case EMotionType::Static:
			return TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Static>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

		default:
			JPH_ASSERT(false);
			break;
		}
		break;

	case EMotionType::Static:
		switch (body2->GetMotionType())
		{
		case EMotionType::Dynamic:
			return TemplatedAddContactConstraint<EMotionType::Static, EMotionType::Dynamic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

		case EMotionType::Kinematic:
			return TemplatedAddContactConstraint<EMotionType::Static, EMotionType::Kinematic>(ioContactAllocator, inBodyPairHandle, *body1, *body2, *manifold);

		case EMotionType::Static: // Static vs static not possible
		default:
			JPH_ASSERT(false);
			break;
		}
		break;

	default:
		JPH_ASSERT(false);
		break;
	}

	return false;
}

void ContactConstraintManager::OnCCDContactAdded(ContactAllocator &ioContactAllocator, const Body &inBody1, const Body &inBody2, const ContactManifold &inManifold, ContactSettings &outSettings)
{
	JPH_ASSERT(inManifold.mWorldSpaceNormal.IsNormalized());

	// Calculate contact settings
	outSettings.mCombinedFriction = mCombineFriction(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	outSettings.mCombinedRestitution = mCombineRestitution(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	outSettings.mIsSensor = false; // For now, no sensors are supported during CCD
	outSettings.mIsSweepPath = true; // CRUMB: the listener builds no soft patch on these poses

	// The remainder of this function only deals with calling contact callbacks, if there's no contact callback we also don't need to do this work
	if (mContactListener != nullptr)
	{
		// Swap bodies so that body 1 id < body 2 id
		const ContactManifold *manifold;
		const Body *body1, *body2;
		ContactManifold temp;
		if (inBody2.GetID() < inBody1.GetID())
		{
			body1 = &inBody2;
			body2 = &inBody1;
			temp = inManifold.SwapShapes();
			manifold = &temp;
		}
		else
		{
			body1 = &inBody1;
			body2 = &inBody2;
			manifold = &inManifold;
		}

		// Calculate hash
		SubShapeIDPair key { body1->GetID(), manifold->mSubShapeID1, body2->GetID(), manifold->mSubShapeID2 };
		uint64 key_hash = key.GetHash();

		// Check if we already created this contact this physics update
		ManifoldCache &write_cache = mCache[mCacheWriteIdx];
		MKVAndCreated new_manifold_kv = write_cache.FindOrCreate(ioContactAllocator, key, key_hash, 0);
		if (new_manifold_kv.second)
		{
			// This contact is new for this physics update, check if previous update we already had this contact.
			const ManifoldCache &read_cache = mCache[mCacheWriteIdx ^ 1];
			const MKeyValue *old_manifold_kv = read_cache.Find(key, key_hash);
			if (old_manifold_kv == nullptr)
			{
				// New contact
				mContactListener->OnContactAdded(*body1, *body2, *manifold, outSettings);
			}
			else
			{
				// Existing contact
				mContactListener->OnContactPersisted(*body1, *body2, *manifold, outSettings);

				// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
				old_manifold_kv->GetValue().mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;
			}

			// Check if the cache is full
			if (new_manifold_kv.first != nullptr)
			{
				// We don't store any contact points in this manifold as it is not for caching impulses, we only need to know that the contact was created
				CachedManifold &new_manifold = new_manifold_kv.first->GetValue();
				new_manifold.mContactNormal = { 0, 0, 0 };
				new_manifold.mFlags |= (uint16)CachedManifold::EFlags::CCDContact;
			}
		}
		else
		{
			// Already found this contact this physics update.
			// Note that we can trigger OnContactPersisted multiple times per physics update, but otherwise we have no way of obtaining the settings
			mContactListener->OnContactPersisted(*body1, *body2, *manifold, outSettings);
		}

		// If we swapped body1 and body2 we need to swap the mass scales back
		if (manifold == &temp)
		{
			std::swap(outSettings.mInvMassScale1, outSettings.mInvMassScale2);
			std::swap(outSettings.mInvInertiaScale1, outSettings.mInvInertiaScale2);
			// Note we do not need to negate the relative surface velocity as it is not applied by the CCD collision constraint
		}
	}

	JPH_ASSERT(outSettings.mIsSensor || !(inBody1.IsSensor() || inBody2.IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
}

void ContactConstraintManager::SortContacts(uint32 *inConstraintIdxBegin, uint32 *inConstraintIdxEnd) const
{
	JPH_PROFILE_FUNCTION();

	QuickSort(inConstraintIdxBegin, inConstraintIdxEnd, [this](uint32 inLHS, uint32 inRHS) {
		const ContactConstraint &lhs = mConstraints[inLHS];
		const ContactConstraint &rhs = mConstraints[inRHS];

		// Most of the time the sort key will be different so we sort on that
		if (lhs.mSortKey != rhs.mSortKey)
			return lhs.mSortKey < rhs.mSortKey;

		// If they're equal we use the IDs of body 1 to order
		if (lhs.mBody1 != rhs.mBody1)
			return lhs.mBody1->GetID() < rhs.mBody1->GetID();

		// If they're still equal we use the IDs of body 2 to order
		if (lhs.mBody2 != rhs.mBody2)
			return lhs.mBody2->GetID() < rhs.mBody2->GetID();

		JPH_ASSERT(inLHS == inRHS, "Hash collision, ordering will be inconsistent");
		return false;
	});
}

void ContactConstraintManager::FinalizeContactCacheAndCallContactPointRemovedCallbacks(uint inExpectedNumBodyPairs, uint inExpectedNumManifolds)
{
	JPH_PROFILE_FUNCTION();

#ifdef JPH_ENABLE_ASSERTS
	// Mark cache as finalized
	ManifoldCache &old_write_cache = mCache[mCacheWriteIdx];
	old_write_cache.Finalize();

	// Check that the count of body pairs and manifolds that we tracked outside of the cache (to avoid contention on an atomic) is correct
	JPH_ASSERT(old_write_cache.GetNumBodyPairs() == inExpectedNumBodyPairs);
	JPH_ASSERT(old_write_cache.GetNumManifolds() == inExpectedNumManifolds);
#endif

	// Buffers are now complete, make write buffer the read buffer
	mCacheWriteIdx ^= 1;

	// Get the old read cache / new write cache
	ManifoldCache &old_read_cache = mCache[mCacheWriteIdx];

	// Call the contact point removal callbacks
	if (mContactListener != nullptr)
		old_read_cache.ContactPointRemovedCallbacks(mContactListener);

	// We're done with the old read cache now
	old_read_cache.Clear();

	// Use the amount of contacts from the last iteration to determine the amount of buckets to use in the hash map for the next iteration
	old_read_cache.Prepare(inExpectedNumBodyPairs, inExpectedNumManifolds);
}

bool ContactConstraintManager::WereBodiesInContact(const BodyID &inBody1ID, const BodyID &inBody2ID) const
{
	// The body pair needs to be in the cache and it needs to have a manifold (otherwise it's just a record indicating that there are no collisions)
	const ManifoldCache &read_cache = mCache[mCacheWriteIdx ^ 1];
	BodyPair key;
	if (inBody1ID < inBody2ID)
		key = BodyPair(inBody1ID, inBody2ID);
	else
		key = BodyPair(inBody2ID, inBody1ID);
	uint64 key_hash = key.GetHash();
	const BPKeyValue *kv = read_cache.Find(key, key_hash);
	return kv != nullptr && kv->GetValue().mFirstCachedManifold != ManifoldMap::cInvalidHandle;
}

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE void ContactConstraintManager::sWarmStartConstraint(ContactConstraint &ioConstraint, MotionProperties *ioMotionProperties1, MotionProperties *ioMotionProperties2, float inWarmStartImpulseRatio)
{
	// Calculate tangents
	Vec3 t1, t2;
	ioConstraint.GetTangents(t1, t2);

	Vec3 ws_normal = ioConstraint.GetWorldSpaceNormal();

	for (WorldContactPoint &wcp : ioConstraint.mContactPoints)
	{
		// Warm starting: Apply impulse from last frame
		if (wcp.mFrictionConstraint1.IsActive() || wcp.mFrictionConstraint2.IsActive())
		{
			wcp.mFrictionConstraint1.TemplatedWarmStart<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, t1, inWarmStartImpulseRatio);
			wcp.mFrictionConstraint2.TemplatedWarmStart<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, t2, inWarmStartImpulseRatio);
		}
		wcp.mNonPenetrationConstraint.TemplatedWarmStart<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, ws_normal, inWarmStartImpulseRatio);
	}
}

template <class MotionPropertiesCallback>
void ContactConstraintManager::WarmStartVelocityConstraints(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd, float inWarmStartImpulseRatio, MotionPropertiesCallback &ioCallback)
{
	JPH_PROFILE_FUNCTION();

	for (const uint32 *constraint_idx = inConstraintIdxBegin; constraint_idx < inConstraintIdxEnd; ++constraint_idx)
	{
		ContactConstraint &constraint = mConstraints[*constraint_idx];

		// Fetch bodies
		Body &body1 = *constraint.mBody1;
		EMotionType motion_type1 = body1.GetMotionType();
		MotionProperties *motion_properties1 = body1.GetMotionPropertiesUnchecked();

		Body &body2 = *constraint.mBody2;
		EMotionType motion_type2 = body2.GetMotionType();
		MotionProperties *motion_properties2 = body2.GetMotionPropertiesUnchecked();

		// CRUMB: a soft patch warm starts its own rows
		if (constraint.mSoftPatchIndex != cNoSoftPatch)
		{
			WarmStartSoftPatch(constraint, inWarmStartImpulseRatio);
			if (motion_type2 == EMotionType::Dynamic)
				ioCallback(motion_properties2);
			if (motion_type1 == EMotionType::Dynamic)
				ioCallback(motion_properties1);
			continue;
		}

		// Dispatch to the correct templated form
		// Note: Warm starting doesn't differentiate between kinematic/static bodies so we handle both as static bodies
		if (motion_type1 == EMotionType::Dynamic)
		{
			if (motion_type2 == EMotionType::Dynamic)
			{
				sWarmStartConstraint<EMotionType::Dynamic, EMotionType::Dynamic>(constraint, motion_properties1, motion_properties2, inWarmStartImpulseRatio);

				ioCallback(motion_properties2);
			}
			else
				sWarmStartConstraint<EMotionType::Dynamic, EMotionType::Static>(constraint, motion_properties1, motion_properties2, inWarmStartImpulseRatio);

			ioCallback(motion_properties1);
		}
		else
		{
			JPH_ASSERT(motion_type2 == EMotionType::Dynamic);

			sWarmStartConstraint<EMotionType::Static, EMotionType::Dynamic>(constraint, motion_properties1, motion_properties2, inWarmStartImpulseRatio);

			ioCallback(motion_properties2);
		}
	}
}

// Specialize for the two body callback types
template void ContactConstraintManager::WarmStartVelocityConstraints<CalculateSolverSteps>(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd, float inWarmStartImpulseRatio, CalculateSolverSteps &ioCallback);
template void ContactConstraintManager::WarmStartVelocityConstraints<DummyCalculateSolverSteps>(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd, float inWarmStartImpulseRatio, DummyCalculateSolverSteps &ioCallback);

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE bool ContactConstraintManager::sSolveVelocityConstraint(ContactConstraint &ioConstraint, MotionProperties *ioMotionProperties1, MotionProperties *ioMotionProperties2)
{
	bool any_impulse_applied = false;

	// Calculate tangents
	Vec3 t1, t2;
	ioConstraint.GetTangents(t1, t2);

	// First apply all friction constraints (non-penetration is more important than friction)
	for (WorldContactPoint &wcp : ioConstraint.mContactPoints)
	{
		// Check if friction is enabled
		if (wcp.mFrictionConstraint1.IsActive() || wcp.mFrictionConstraint2.IsActive())
		{
			// Calculate impulse to stop motion in tangential direction
			float lambda1 = wcp.mFrictionConstraint1.TemplatedSolveVelocityConstraintGetTotalLambda<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, t1);
			float lambda2 = wcp.mFrictionConstraint2.TemplatedSolveVelocityConstraintGetTotalLambda<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, t2);
			float total_lambda_sq = Square(lambda1) + Square(lambda2);

			// Calculate max impulse that can be applied. Note that we're using the non-penetration impulse from the previous iteration here.
			// We do this because non-penetration is more important so is solved last (the last things that are solved in an iterative solver
			// contribute the most).
			float max_lambda_f = ioConstraint.mCombinedFriction * wcp.mNonPenetrationConstraint.GetTotalLambda();

			// If the total lambda that we will apply is too large, scale it back
			if (total_lambda_sq > Square(max_lambda_f))
			{
				float scale = max_lambda_f / sqrt(total_lambda_sq);
				lambda1 *= scale;
				lambda2 *= scale;
			}

			// Apply the friction impulse
			if (wcp.mFrictionConstraint1.TemplatedSolveVelocityConstraintApplyLambda<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, t1, lambda1))
				any_impulse_applied = true;
			if (wcp.mFrictionConstraint2.TemplatedSolveVelocityConstraintApplyLambda<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, t2, lambda2))
				any_impulse_applied = true;
		}
	}

	Vec3 ws_normal = ioConstraint.GetWorldSpaceNormal();

	// Then apply all non-penetration constraints
	for (WorldContactPoint &wcp : ioConstraint.mContactPoints)
	{
		// Solve non penetration velocities
		if (wcp.mNonPenetrationConstraint.TemplatedSolveVelocityConstraint<Type1, Type2>(ioMotionProperties1, ioConstraint.mInvMass1, ioMotionProperties2, ioConstraint.mInvMass2, ws_normal, 0.0f, FLT_MAX))
			any_impulse_applied = true;
	}

	return any_impulse_applied;
}

bool ContactConstraintManager::SolveVelocityConstraints(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd)
{
	JPH_PROFILE_FUNCTION();

	bool any_impulse_applied = false;

	for (const uint32 *constraint_idx = inConstraintIdxBegin; constraint_idx < inConstraintIdxEnd; ++constraint_idx)
	{
		ContactConstraint &constraint = mConstraints[*constraint_idx];

		// Fetch bodies
		Body &body1 = *constraint.mBody1;
		EMotionType motion_type1 = body1.GetMotionType();
		MotionProperties *motion_properties1 = body1.GetMotionPropertiesUnchecked();

		Body &body2 = *constraint.mBody2;
		EMotionType motion_type2 = body2.GetMotionType();
		MotionProperties *motion_properties2 = body2.GetMotionPropertiesUnchecked();

		// CRUMB: a soft patch solves its own rows
		if (constraint.mSoftPatchIndex != cNoSoftPatch)
		{
			any_impulse_applied |= SolveSoftPatch(constraint);
			continue;
		}

		// Dispatch to the correct templated form
		switch (motion_type1)
		{
		case EMotionType::Dynamic:
			switch (motion_type2)
			{
			case EMotionType::Dynamic:
				any_impulse_applied |= sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Dynamic>(constraint, motion_properties1, motion_properties2);
				break;

			case EMotionType::Kinematic:
				any_impulse_applied |= sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Kinematic>(constraint, motion_properties1, motion_properties2);
				break;

			case EMotionType::Static:
				any_impulse_applied |= sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Static>(constraint, motion_properties1, motion_properties2);
				break;

			default:
				JPH_ASSERT(false);
				break;
			}
			break;

		case EMotionType::Kinematic:
			JPH_ASSERT(motion_type2 == EMotionType::Dynamic);
			any_impulse_applied |= sSolveVelocityConstraint<EMotionType::Kinematic, EMotionType::Dynamic>(constraint, motion_properties1, motion_properties2);
			break;

		case EMotionType::Static:
			JPH_ASSERT(motion_type2 == EMotionType::Dynamic);
			any_impulse_applied |= sSolveVelocityConstraint<EMotionType::Static, EMotionType::Dynamic>(constraint, motion_properties1, motion_properties2);
			break;

		default:
			JPH_ASSERT(false);
			break;
		}
	}

	return any_impulse_applied;
}

void ContactConstraintManager::StoreAppliedImpulses(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd) const
{
	// Copy back total applied impulse to cache for the next frame
	for (const uint32 *constraint_idx = inConstraintIdxBegin; constraint_idx < inConstraintIdxEnd; ++constraint_idx)
	{
		const ContactConstraint &constraint = mConstraints[*constraint_idx];

		// CRUMB: a soft patch writes its tail (and its report)
		if (constraint.mSoftPatchIndex != cNoSoftPatch)
		{
			StoreSoftPatch(constraint);
			continue;
		}

		for (const WorldContactPoint &wcp : constraint.mContactPoints)
		{
			wcp.mContactPoint->mNonPenetrationLambda = wcp.mNonPenetrationConstraint.GetTotalLambda();
			wcp.mContactPoint->mFrictionLambda[0] = wcp.mFrictionConstraint1.GetTotalLambda();
			wcp.mContactPoint->mFrictionLambda[1] = wcp.mFrictionConstraint2.GetTotalLambda();
		}

		// CRUMB: report what the solve applied, for a manifold that asked
		if (constraint.mReportSolved && mContactListener != nullptr && !constraint.mContactPoints.empty())
			ReportSolvedContact(constraint);
	}
}

void ContactConstraintManager::ReportSolvedContact(const ContactConstraint &inConstraint) const
{
	const Body &body1 = *inConstraint.mBody1;
	const Body &body2 = *inConstraint.mBody2;

	// The velocity solve does not move the bodies, so each point is rebuilt from the cached
	// body-local positions exactly as the solve placed it: the midpoint of the two contact points.
	RMat44 transform1 = body1.GetCenterOfMassTransform();
	RMat44 transform2 = body2.GetCenterOfMassTransform();
	Vec3 normal = inConstraint.GetWorldSpaceNormal();
	Vec3 t1, t2;
	inConstraint.GetTangents(t1, t2);

	RVec3 points[MaxContactPoints];
	Vec3 impulses[MaxContactPoints];
	uint count = 0;
	Vec3 linear = Vec3::sZero();
	float normal_sum = 0.0f;
	float friction_sum = 0.0f;
	RVec3 weighted = RVec3::sZero();
	RVec3 plain = RVec3::sZero();
	for (const WorldContactPoint &wcp : inConstraint.mContactPoints)
	{
		RVec3 p1 = transform1 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition1);
		RVec3 p2 = transform2 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition2);
		RVec3 p = 0.5_r * (p1 + p2);
		float lambda_n = max(wcp.mNonPenetrationConstraint.GetTotalLambda(), 0.0f);
		float lambda_f1 = wcp.mFrictionConstraint1.GetTotalLambda();
		float lambda_f2 = wcp.mFrictionConstraint2.GetTotalLambda();
		Vec3 impulse = lambda_n * normal + lambda_f1 * t1 + lambda_f2 * t2;
		points[count] = p;
		impulses[count] = impulse;
		++count;
		linear += impulse;
		normal_sum += lambda_n;
		friction_sum += sqrt(Square(lambda_f1) + Square(lambda_f2));
		weighted += Real(lambda_n) * p;
		plain += p;
	}

	ContactSolvedReport report;
	report.mWorldNormal = normal;
	report.mCentre = normal_sum > 0.0f? weighted / Real(normal_sum) : plain / Real(count);
	report.mLinearImpulse = linear;
	Vec3 angular = Vec3::sZero();
	for (uint i = 0; i < count; ++i)
		angular += Vec3(points[i] - report.mCentre).Cross(impulses[i]);
	report.mAngularImpulse = angular;
	report.mNormalImpulse = normal_sum;
	report.mArea = 0.0f;

	// The solve clamps each point's friction to the combined friction times its pressing impulse, so
	// the summed magnitudes over the summed limits is 1 exactly when every point slides. Summed rather
	// than the largest point's share: a rigid manifold's split between its points is not determined,
	// and one lightly pressed point can reach its limit while the manifold holds.
	float limit = inConstraint.mCombinedFriction * normal_sum;
	report.mFrictionUse = limit > 0.0f? min(friction_sum / limit, 1.0f) : (friction_sum > 0.0f? 1.0f : 0.0f);

	SubShapeIDPair pair(body1.GetID(), inConstraint.mSubShapeID1, body2.GetID(), inConstraint.mSubShapeID2);
	mContactListener->OnContactSolved(body1, body2, pair, report);
}

bool ContactConstraintManager::SolvePositionConstraints(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd)
{
	JPH_PROFILE_FUNCTION();

	bool any_impulse_applied = false;

	for (const uint32 *constraint_idx = inConstraintIdxBegin; constraint_idx < inConstraintIdxEnd; ++constraint_idx)
	{
		ContactConstraint &constraint = mConstraints[*constraint_idx];

		// CRUMB: a soft patch corrects only its core guard (and a first touch solved as an impact)
		if (constraint.mSoftPatchIndex != cNoSoftPatch)
		{
			if (SolveSoftPatchPosition(constraint))
				any_impulse_applied = true;
			continue;
		}

		// Fetch bodies
		Body &body1 = *constraint.mBody1;
		Body &body2 = *constraint.mBody2;

		// Get transforms
		RMat44 transform1 = body1.GetCenterOfMassTransform();
		RMat44 transform2 = body2.GetCenterOfMassTransform();

		Vec3 ws_normal = constraint.GetWorldSpaceNormal();

		for (WorldContactPoint &wcp : constraint.mContactPoints)
		{
			// Calculate new contact point positions in world space (the bodies may have moved)
			RVec3 p1 = transform1 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition1);
			RVec3 p2 = transform2 * Vec3::sLoadFloat3Unsafe(wcp.mContactPoint->mPosition2);

			// Calculate separation along the normal (negative if interpenetrating)
			// Allow a little penetration by default (PhysicsSettings::mPenetrationSlop) to avoid jittering between contact/no-contact which wipes out the contact cache and warm start impulses
			// Clamp penetration to a max PhysicsSettings::mMaxPenetrationDistance so that we don't apply a huge impulse if we're penetrating a lot
			float separation = max(Vec3(p2 - p1).Dot(ws_normal) + mPhysicsSettings.mPenetrationSlop, -mPhysicsSettings.mMaxPenetrationDistance);

			// Only enforce constraint when separation < 0 (otherwise we're apart)
			if (separation < 0.0f)
			{
				// Update constraint properties (bodies may have moved)
				wcp.CalculateNonPenetrationConstraintProperties(body1, constraint.mInvMass1, constraint.mInvInertiaScale1, body2, constraint.mInvMass2, constraint.mInvInertiaScale2, p1, p2, ws_normal);

				// Solve position errors
				if (wcp.mNonPenetrationConstraint.SolvePositionConstraintWithMassOverride(body1, constraint.mInvMass1, body2, constraint.mInvMass2, ws_normal, separation, mPhysicsSettings.mBaumgarte))
					any_impulse_applied = true;
			}
		}
	}

	return any_impulse_applied;
}

void ContactConstraintManager::RecycleConstraintBuffer()
{
	// Reset constraint array
	mNumConstraints = 0;
	mNumSoftPatches = 0; // CRUMB
}

void ContactConstraintManager::FinishConstraintBuffer()
{
	// CRUMB: the soft patches' side table was allocated after the constraints, so it is freed first
	if (mSoftPatches != nullptr)
	{
		mUpdateContext->mTempAllocator->Free(mSoftPatches, mMaxSoftPatches * sizeof(SoftPatchConstraint));
		mSoftPatches = nullptr;
	}

	// Free constraints buffer
	mUpdateContext->mTempAllocator->Free(mConstraints, mMaxConstraints * sizeof(ContactConstraint));
	mConstraints = nullptr;
	mNumConstraints = 0;

	// Reset update context
	mUpdateContext = nullptr;
}

void ContactConstraintManager::SaveState(StateRecorder &inStream, const StateRecorderFilter *inFilter) const
{
	mCache[mCacheWriteIdx ^ 1].SaveState(inStream, inFilter);
}

bool ContactConstraintManager::RestoreState(StateRecorder &inStream, const StateRecorderFilter *inFilter)
{
	bool success = mCache[mCacheWriteIdx].RestoreState(mCache[mCacheWriteIdx ^ 1], inStream, inFilter);

	// If this is the last part, the cache is finalized
	if (inStream.IsLastPart())
	{
		mCacheWriteIdx ^= 1;
		mCache[mCacheWriteIdx].Clear();
	}

	return success;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// CRUMB: the compliant contact (soft patches)
//
// A manifold whose listener built a SoftContactPatch is solved as a few rows in place of its contact
// points: a normal row at the centre of pressure that applies the patch law's force at the start-of-step
// squash explicitly (h F) and adds an implicit share B = c + theta k h on the closing speed, a core guard
// at the deepest point (a rigid speculative row against the backing), tilt rows (compliant, about in-plane
// axes through the centre), a friction block (two tangents and the spin, compliant pre-slip springs or a
// rigid stick, limited together by the soft-finger ellipse) and rolling resistance rows (rigid, saturating
// at the rolling arm times the normal impulse). theta comes from the pair-sum bound over each island's
// soft rows (FinalizeSoftPatches), the rows are warm started from last step's impulses and then corrected
// together per soft cluster (WarmStartSoftClusters), and the episode rule decides a first touch the step
// cannot resolve as an impact with the material's restitution.
////////////////////////////////////////////////////////////////////////////////////////////////////////

// A compliant row whose h B w is below this is applied as a fixed impulse (no damping and no implicit share)
static constexpr float cSoftFixedEpsilon = 1.0e-6f;

// The block warm start solves at most this many compliant rows together; a larger cluster patch by patch
static constexpr int cSoftBlockRowCap = 64;

// The most soft patches one island's clusters are formed over, and the most dynamic bodies its pair sums
// keep apart; beyond them a patch is its own cluster and a row falls back to its own bound
static constexpr int cSoftMaxClusterPatches = 256;
static constexpr int cSoftMaxSumBodies = 512;

// Bisections of the soft-finger projection's multiplier (a fixed count, so the solve repeats bit for bit)
static constexpr int cSoftEllipseBisections = 48;

// The smallest spin radius the ellipse divides by (mm in CRUMB's units: one micrometre)
static constexpr double cSoftMinSpinRadius = 1.0e-3;

// The two bodies of a soft patch as its rows see them
struct SoftPatchBodies
{
	MotionProperties *		mWrite1 = nullptr;			// set for a dynamic body: the rows change its velocity
	MotionProperties *		mWrite2 = nullptr;
	const MotionProperties *mRead1 = nullptr;			// set unless the body is static: the rows read its velocity
	const MotionProperties *mRead2 = nullptr;
	float					mInvMass1 = 0.0f;			// the contact's inverse masses, zero unless dynamic
	float					mInvMass2 = 0.0f;
};

static inline SoftPatchBodies sSoftPatchBodies(Body &ioBody1, float inInvMass1, Body &ioBody2, float inInvMass2)
{
	SoftPatchBodies b;
	if (ioBody1.IsDynamic())
	{
		b.mWrite1 = ioBody1.GetMotionPropertiesUnchecked();
		b.mInvMass1 = inInvMass1;
	}
	if (!ioBody1.IsStatic())
		b.mRead1 = ioBody1.GetMotionPropertiesUnchecked();
	if (ioBody2.IsDynamic())
	{
		b.mWrite2 = ioBody2.GetMotionPropertiesUnchecked();
		b.mInvMass2 = inInvMass2;
	}
	if (!ioBody2.IsStatic())
		b.mRead2 = ioBody2.GetMotionPropertiesUnchecked();
	return b;
}

// J v of a row: the opening rate (linear rows) or the relative spin about the axis (angular rows)
template <class Row>
static inline float sSoftRowVelocity(const Row &inRow, const SoftPatchBodies &inB)
{
	float jv = 0.0f;
	if (inB.mRead2 != nullptr)
		jv += inRow.mLinear.Dot(inB.mRead2->GetLinearVelocity()) + inRow.mAngular2.Dot(inB.mRead2->GetAngularVelocity());
	if (inB.mRead1 != nullptr)
		jv -= inRow.mLinear.Dot(inB.mRead1->GetLinearVelocity()) + inRow.mAngular1.Dot(inB.mRead1->GetAngularVelocity());
	return jv;
}

// Apply an impulse inLambda along a row
template <class Row>
static inline void sSoftRowApply(const Row &inRow, const SoftPatchBodies &inB, float inLambda)
{
	if (inLambda == 0.0f)
		return;
	if (inB.mWrite1 != nullptr)
	{
		inB.mWrite1->SubLinearVelocityStep((inLambda * inB.mInvMass1) * inRow.mLinear);
		inB.mWrite1->SubAngularVelocityStep(inLambda * inRow.mInvIAngular1);
	}
	if (inB.mWrite2 != nullptr)
	{
		inB.mWrite2->AddLinearVelocityStep((inLambda * inB.mInvMass2) * inRow.mLinear);
		inB.mWrite2->AddAngularVelocityStep(inLambda * inRow.mInvIAngular2);
	}
}

template <class Row>
static inline void sSoftRowReset(Row &outRow)
{
	outRow.mLinear = Vec3::sZero();
	outRow.mAngular1 = Vec3::sZero();
	outRow.mAngular2 = Vec3::sZero();
	outRow.mInvIAngular1 = Vec3::sZero();
	outRow.mInvIAngular2 = Vec3::sZero();
	outRow.mInvEffectiveMass = 0.0f;
	outRow.mEffectiveMass = 0.0f;
	outRow.mSoftness = 0.0f;
	outRow.mBias = 0.0f;
	outRow.mTotalLambda = 0.0f;
	outRow.mStiffness = 0.0f;
	outRow.mDampingRatio = 0.0f;
	outRow.mPreload = 0.0f;
	outRow.mTheta = 0.0f;
	outRow.mMode = ContactConstraintManager::SoftRowInactive;
}

// A linear row along inAxis at the point r1 from body 1's centre of mass and r2 from body 2's
template <class Row>
static inline void sSoftRowLinear(Row &ioRow, Vec3Arg inAxis, Vec3Arg inR1, Vec3Arg inR2, float inInvMass1, Mat44Arg inInvI1, float inInvMass2, Mat44Arg inInvI2)
{
	ioRow.mLinear = inAxis;
	ioRow.mAngular1 = inR1.Cross(inAxis);
	ioRow.mAngular2 = inR2.Cross(inAxis);
	ioRow.mInvIAngular1 = inInvI1.Multiply3x3(ioRow.mAngular1);
	ioRow.mInvIAngular2 = inInvI2.Multiply3x3(ioRow.mAngular2);
	ioRow.mInvEffectiveMass = inInvMass1 + inInvMass2 + ioRow.mAngular1.Dot(ioRow.mInvIAngular1) + ioRow.mAngular2.Dot(ioRow.mInvIAngular2);
}

// An angular row about inAxis
template <class Row>
static inline void sSoftRowAngular(Row &ioRow, Vec3Arg inAxis, Mat44Arg inInvI1, Mat44Arg inInvI2)
{
	ioRow.mLinear = Vec3::sZero();
	ioRow.mAngular1 = inAxis;
	ioRow.mAngular2 = inAxis;
	ioRow.mInvIAngular1 = inInvI1.Multiply3x3(inAxis);
	ioRow.mInvIAngular2 = inInvI2.Multiply3x3(inAxis);
	ioRow.mInvEffectiveMass = inAxis.Dot(ioRow.mInvIAngular1) + inAxis.Dot(ioRow.mInvIAngular2);
}

// A velocity row with a bias, or no row when nothing can move along it
template <class Row>
static inline void sSoftRowRigid(Row &ioRow, float inBias)
{
	if (ioRow.mInvEffectiveMass > 0.0f)
	{
		ioRow.mMode = ContactConstraintManager::SoftRowRigid;
		ioRow.mEffectiveMass = 1.0f / ioRow.mInvEffectiveMass;
		ioRow.mSoftness = 0.0f;
		ioRow.mBias = inBias;
	}
	else
		ioRow.mMode = ContactConstraintManager::SoftRowInactive;
}

// A compliant row: its law's stiffness, damping ratio and explicit force (its softness is set per island)
template <class Row>
static inline void sSoftRowCompliant(Row &ioRow, float inStiffness, float inDampingRatio, float inPreload)
{
	if (ioRow.mInvEffectiveMass > 0.0f && inStiffness > 0.0f)
	{
		ioRow.mMode = ContactConstraintManager::SoftRowCompliant;
		ioRow.mStiffness = inStiffness;
		ioRow.mDampingRatio = max(inDampingRatio, 0.0f);
		ioRow.mPreload = inPreload;
	}
	else
		ioRow.mMode = ContactConstraintManager::SoftRowInactive;
}

// One sweep of a row (rigid or compliant) with its total clamped to [inMin, inMax]; returns the impulse applied
template <class Row>
static inline float sSoftRowSolve(Row &ioRow, const SoftPatchBodies &inB, float inMin, float inMax)
{
	float jv = sSoftRowVelocity(ioRow, inB);
	float lambda = -ioRow.mEffectiveMass * (jv + ioRow.mBias + ioRow.mSoftness * ioRow.mTotalLambda);
	float total = Clamp(ioRow.mTotalLambda + lambda, inMin, inMax);
	lambda = total - ioRow.mTotalLambda;
	ioRow.mTotalLambda = total;
	sSoftRowApply(ioRow, inB, lambda);
	return lambda;
}

// J_i M^-1 J_j^T for two rows on the same two bodies
template <class Row>
static inline double sSoftRowCoupling(const Row &inI, const Row &inJ, const SoftPatchBodies &inB)
{
	return double(inB.mInvMass1 + inB.mInvMass2) * double(inI.mLinear.Dot(inJ.mLinear))
		+ double(inI.mAngular1.Dot(inJ.mInvIAngular1)) + double(inI.mAngular2.Dot(inJ.mInvIAngular2));
}

// Solve A x = b in place (A is n x n, row major) by Gaussian elimination with partial pivoting, in a fixed
// order; returns false (x zero) when A is singular
static bool sSoftSolveDense(double *ioA, double *ioB, int inN, double *outX)
{
	for (int col = 0; col < inN; ++col)
	{
		int pivot = col;
		double best = abs(ioA[col * inN + col]);
		for (int row = col + 1; row < inN; ++row)
		{
			double v = abs(ioA[row * inN + col]);
			if (v > best)
			{
				best = v;
				pivot = row;
			}
		}
		if (!(best > 0.0))
		{
			for (int i = 0; i < inN; ++i)
				outX[i] = 0.0;
			return false;
		}
		if (pivot != col)
		{
			for (int k = 0; k < inN; ++k)
				std::swap(ioA[col * inN + k], ioA[pivot * inN + k]);
			std::swap(ioB[col], ioB[pivot]);
		}
		double inv = 1.0 / ioA[col * inN + col];
		for (int row = col + 1; row < inN; ++row)
		{
			double f = ioA[row * inN + col] * inv;
			if (f == 0.0)
				continue;
			for (int k = col; k < inN; ++k)
				ioA[row * inN + k] -= f * ioA[col * inN + k];
			ioB[row] -= f * ioB[col];
		}
	}
	for (int row = inN - 1; row >= 0; --row)
	{
		double s = ioB[row];
		for (int k = row + 1; k < inN; ++k)
			s -= ioA[row * inN + k] * outX[k];
		outX[row] = s / ioA[row * inN + row];
	}
	return true;
}

// The soft-finger limit (t1 / (mu L))^2 + (t2 / (mu L))^2 + (spin / (mu L r_s))^2 <= 1 on the friction block's
// total impulse. Inside it the impulse is unchanged. Outside it, the maximal-dissipation step in the block's
// metric K: l = (K + nu D)^-1 K l*, with the multiplier nu found by a fixed count of bisections in log space.
// Returns true when it projected (the block slides).
static bool sSoftFingerProject(const double inK[9], double ioLambda[3], double inMuL, double inSpinRadius)
{
	if (!(inMuL > 0.0))
	{
		ioLambda[0] = ioLambda[1] = ioLambda[2] = 0.0;
		return true;
	}
	double r = max(inSpinRadius, cSoftMinSpinRadius);
	double D[3] = { 1.0 / (inMuL * inMuL), 1.0 / (inMuL * inMuL), 1.0 / ((inMuL * r) * (inMuL * r)) };
	double q = ioLambda[0] * ioLambda[0] * D[0] + ioLambda[1] * ioLambda[1] * D[1] + ioLambda[2] * ioLambda[2] * D[2];
	if (q <= 1.0)
		return false;
	double kl[3];
	for (int i = 0; i < 3; ++i)
		kl[i] = inK[3 * i] * ioLambda[0] + inK[3 * i + 1] * ioLambda[1] + inK[3 * i + 2] * ioLambda[2];
	double scale = (inK[0] + inK[4] + inK[8]) / (D[0] + D[1] + D[2]);
	double lo = -40.0, hi = 40.0;
	double x[3] = { 0.0, 0.0, 0.0 };
	for (int it = 0; it <= cSoftEllipseBisections; ++it)
	{
		bool last = it == cSoftEllipseBisections;
		double mid = last? hi : 0.5 * (lo + hi);
		double nu = std::exp(mid) * scale;
		double A[9];
		for (int i = 0; i < 9; ++i)
			A[i] = inK[i];
		A[0] += nu * D[0];
		A[4] += nu * D[1];
		A[8] += nu * D[2];
		double b[3] = { kl[0], kl[1], kl[2] };
		sSoftSolveDense(A, b, 3, x);
		if (last)
			break;
		if (x[0] * x[0] * D[0] + x[1] * x[1] * D[1] + x[2] * x[2] * D[2] > 1.0)
			lo = mid;
		else
			hi = mid;
	}
	ioLambda[0] = x[0];
	ioLambda[1] = x[1];
	ioLambda[2] = x[2];
	return true;
}

void ContactConstraintManager::SetupSoftPatch(ContactConstraint &ioConstraint, SoftPatchConstraint &outPatch, const SoftContactPatch &inPatch, SoftContactTail &ioTail, bool inWarm, float inInvM1, Mat44Arg inInvI1, float inInvM2, Mat44Arg inInvI2, float inDeltaTime, Vec3Arg inGravity)
{
	Body &body1 = *ioConstraint.mBody1;
	Body &body2 = *ioConstraint.mBody2;
	SoftPatchConstraint &p = *new (&outPatch) SoftPatchConstraint();
	SoftPatchBodies b = sSoftPatchBodies(body1, inInvM1, body2, inInvM2);

	sSoftRowReset(p.mNormal);
	sSoftRowReset(p.mCore);
	for (SoftRow &r : p.mFriction)
		sSoftRowReset(r);
	for (SoftRow &r : p.mTilt)
		sSoftRowReset(r);
	for (SoftRow &r : p.mRoll)
		sSoftRowReset(r);

	Vec3 n = ioConstraint.GetWorldSpaceNormal();
	Vec3 t1 = inPatch.mTangent1;
	Vec3 t2 = inPatch.mTangent2;
	p.mNormalAxis = n;
	p.mTangent1 = t1;
	p.mTangent2 = t2;
	p.mCentre = inPatch.mCentre;
	p.mTail = &ioTail;
	p.mDeltaTime = inDeltaTime;
	p.mSquash = inPatch.mSquash;
	p.mLayer = inPatch.mCoreGap + inPatch.mSquash;
	p.mNumTilt = uint8(min<int>(inPatch.mNumTilt, 2));
	p.mNumRoll = uint8(min<int>(inPatch.mNumRoll, 2));
	for (int i = 0; i < 2; ++i)
	{
		p.mTiltArmNeg[i] = max(inPatch.mTiltArmNeg[i], 0.0f);
		p.mTiltArmPos[i] = max(inPatch.mTiltArmPos[i], 0.0f);
		p.mRollArm[i] = max(inPatch.mRollArm[i], 0.0f);
	}
	p.mSpinRadius = max(inPatch.mSpinRadius, 0.0f);
	p.mStaticFriction = max(inPatch.mStaticFriction, 0.0f);
	p.mSlidingFriction = Clamp(inPatch.mSlidingFriction, 0.0f, p.mStaticFriction);
	p.mShearStiffness = max(inPatch.mShearStiffness, 0.0f);
	p.mSpinStiffness = max(inPatch.mSpinStiffness, 0.0f);
	p.mRelaxationLength = max(inPatch.mRelaxationLength, 0.0f);
	p.mRollingRadius = max(inPatch.mRollingRadius, 0.0f);
	p.mArea = max(inPatch.mArea, 0.0f);
	p.mClass = inPatch.mClass;
	p.mImpact = false;
	p.mDecided = false;
	p.mGap = inPatch.mSquash <= 0.0f;
	p.mRigidStick = inPatch.mRigidStick || p.mShearStiffness <= 0.0f;
	p.mFrictionFixed = false;
	p.mSliding = false;
	p.mWarm = inWarm;
	for (double &k : p.mFrictionMatrix)
		k = 0.0;

	// The deepest point on each surface, kept in each body's frame for the position pass
	Vec3 deep_offset = (0.5f * inPatch.mSquash) * n;
	Vec3(body1.GetInverseCenterOfMassTransform() * (inPatch.mDeepest + deep_offset)).StoreFloat3(&p.mDeepLocal1);
	Vec3(body2.GetInverseCenterOfMassTransform() * (inPatch.mDeepest - deep_offset)).StoreFloat3(&p.mDeepLocal2);

	++mSoftStats.mPatches;

	RVec3 com1 = body1.GetCenterOfMassPosition();
	RVec3 com2 = body2.GetCenterOfMassPosition();
	Vec3 r1 = Vec3(inPatch.mCentre - com1);
	Vec3 r2 = Vec3(inPatch.mCentre - com2);
	Vec3 rd1 = Vec3(inPatch.mDeepest - com1);
	Vec3 rd2 = Vec3(inPatch.mDeepest - com2);

	// The carried pre-slip deflection on this step's tangents (the tail keeps it in body 2's frame)
	if (inWarm)
	{
		Vec3 shear = body2.GetRotation() * Vec3::sLoadFloat3Unsafe(ioTail.mShear);
		p.mShear[0] = shear.Dot(t1);
		p.mShear[1] = shear.Dot(t2);
		p.mSpin = ioTail.mSpinDeflection;
	}
	else
	{
		p.mShear[0] = p.mShear[1] = 0.0f;
		p.mSpin = 0.0f;
	}

	// The approach at the deepest point (velocities already carry this step's gravity and forces), and the part of
	// it this step's gravity and forces made: a first touch is not an impact on that alone. Gravity reaches a body
	// through its gravity factor, or through its accumulated force when the host applies gravity itself (CRUMB's
	// host keeps every gravity factor at 0), so both are read, as the restitution below and the rigid contact's
	// restitution read them. Between two dynamic bodies under the same gravity the two cancel; against a static or
	// kinematic support the whole increment counts.
	Vec3 v1 = b.mRead1 != nullptr? b.mRead1->GetPointVelocityCOM(rd1) : Vec3::sZero();
	Vec3 v2 = b.mRead2 != nullptr? b.mRead2->GetPointVelocityCOM(rd2) : Vec3::sZero();
	float normal_velocity = (v2 - v1).Dot(n);
	float approach = -normal_velocity;
	Vec3 relative_acceleration = Vec3::sZero();
	if (b.mWrite2 != nullptr)
		relative_acceleration += inGravity * b.mWrite2->GetGravityFactor() + b.mWrite2->GetAccumulatedForce() * b.mWrite2->GetInverseMass();
	if (b.mWrite1 != nullptr)
		relative_acceleration -= inGravity * b.mWrite1->GetGravityFactor() + b.mWrite1->GetAccumulatedForce() * b.mWrite1->GetInverseMass();
	float force_closing = max(0.0f, -relative_acceleration.Dot(n) * inDeltaTime);

	// The core guard: a rigid speculative row at the deepest point against the backing
	sSoftRowLinear(p.mCore, n, rd1, rd2, inInvM1, inInvI1, inInvM2, inInvI2);
	sSoftRowRigid(p.mCore, max(0.0f, inPatch.mCoreGap / inDeltaTime));

	// First touch, the episode rule (SOFT-CONTACT brief 5.5): a patch that will not touch this step carries no
	// rows but the core guard; the first step of the episode that touches decides once whether the step can
	// resolve it; for the rest of the episode the compliant rows act on the predicted overlap.
	float d0 = inPatch.mSquash;
	bool touches = d0 > 0.0f || d0 + approach * inDeltaTime > 0.0f;
	if (!touches)
		return; // the tail's contact set and the episode carry on unchanged

	sSoftRowLinear(p.mNormal, n, r1, r2, inInvM1, inInvI1, inInvM2, inInvI2);
	if ((ioTail.mFlags & SoftContactTail::OnsetPending) != 0)
	{
		p.mDecided = true;
		ioTail.mFlags &= ~uint32(SoftContactTail::OnsetPending);

		// The stiffness at the predicted peak squash: the closing energy put into the law from first touch
		float w = p.mNormal.mInvEffectiveMass;
		float stiffness_peak = inPatch.mNormalStiffness;
		float c_law = inPatch.mLawCoefficient;
		float n_law = max(inPatch.mLawExponent, 1.0f);
		if (approach > 0.0f && w > 0.0f && c_law > 0.0f)
		{
			float energy = 0.5f * approach * approach / w;
			float s_peak = std::pow((n_law + 1.0f) * energy / c_law, 1.0f / (n_law + 1.0f));
			stiffness_peak = n_law * c_law * std::pow(s_peak, n_law - 1.0f);
		}
		float a = Square(inDeltaTime) * stiffness_peak * w;
		if (a > 0.5f && approach - force_closing > mPhysicsSettings.mMinVelocityForRestitution)
		{
			p.mImpact = true;
			++mSoftStats.mImpacts;
			if (d0 > 0.0f)
				++mSoftStats.mRestingImpacts;
		}
	}

	if (p.mImpact)
	{
		// An impact the step cannot resolve: the rigid row with the material's restitution, at the deepest
		// point, as the rigid contact sets it up; any overlap goes to the position pass
		sSoftRowLinear(p.mNormal, n, rd1, rd2, inInvM1, inInvI1, inInvM2, inInvI2);
		float speculative = max(0.0f, -d0 / inDeltaTime);
		float bias = speculative;
		if (inPatch.mRestitution > 0.0f && normal_velocity < -mPhysicsSettings.mMinVelocityForRestitution && normal_velocity < -speculative)
		{
			// The step's gravity and forces taken off the bounce, as the rigid contact's restitution does
			float force_delta_velocity = min(0.0f, relative_acceleration.Dot(n) * inDeltaTime);
			bias = inPatch.mRestitution * (normal_velocity - force_delta_velocity);
		}
		sSoftRowRigid(p.mNormal, bias);
	}
	else
	{
		// The normal row's stiffness: the builder's, or, when the builder found nothing pressing (a Line or Point
		// across a gap carries no force and no stiffness of its own), the patch's own law F = C s^n at the predicted
		// overlap s = d0 + u h (the episode rule's implicit share acts on it). The secant C s^(n-1) makes that share
		// carry the law's force at s, k s = C s^n; a linear law gives C itself.
		float stiffness = inPatch.mNormalStiffness;
		if (!(stiffness > 0.0f) && !(inPatch.mForce > 0.0f) && inPatch.mLawCoefficient > 0.0f)
		{
			float n_law = max(inPatch.mLawExponent, 1.0f);
			float s_pred = max(d0 + approach * inDeltaTime, d0);
			if (n_law == 1.0f)
				stiffness = inPatch.mLawCoefficient;
			else if (s_pred > 0.0f)
				stiffness = inPatch.mLawCoefficient * std::pow(s_pred, n_law - 1.0f);
		}
		sSoftRowCompliant(p.mNormal, stiffness, inPatch.mNormalDampingRatio, max(inPatch.mForce, 0.0f));
	}

	// The friction block: two tangents at the centre and the spin about the normal
	if (p.mStaticFriction > 0.0f)
	{
		sSoftRowLinear(p.mFriction[0], t1, r1, r2, inInvM1, inInvI1, inInvM2, inInvI2);
		sSoftRowLinear(p.mFriction[1], t2, r1, r2, inInvM1, inInvI1, inInvM2, inInvI2);
		sSoftRowAngular(p.mFriction[2], n, inInvI1, inInvI2);
		if (p.mImpact || p.mRigidStick)
		{
			for (SoftRow &r : p.mFriction)
				sSoftRowRigid(r, 0.0f);
		}
		else
		{
			sSoftRowCompliant(p.mFriction[0], p.mShearStiffness, inPatch.mDampingRatio, -p.mShearStiffness * p.mShear[0]);
			sSoftRowCompliant(p.mFriction[1], p.mShearStiffness, inPatch.mDampingRatio, -p.mShearStiffness * p.mShear[1]);
			if (p.mSpinStiffness > 0.0f)
				sSoftRowCompliant(p.mFriction[2], p.mSpinStiffness, inPatch.mDampingRatio, -p.mSpinStiffness * p.mSpin);
			else
				sSoftRowRigid(p.mFriction[2], 0.0f);
		}
	}

	if (!p.mImpact)
	{
		// Tilt rows: compliant about in-plane axes through the centre, no preload (the explicit force already acts at
		// the centre of the tilted field)
		for (int i = 0; i < p.mNumTilt; ++i)
		{
			sSoftRowAngular(p.mTilt[i], inPatch.mTiltAxis[i], inInvI1, inInvI2);
			sSoftRowCompliant(p.mTilt[i], inPatch.mTiltStiffness[i], inPatch.mDampingRatio, 0.0f);
		}

		// Rolling resistance rows: a saturating damper (rigid, within the arm times the normal impulse)
		for (int i = 0; i < p.mNumRoll; ++i)
		{
			if (p.mRollArm[i] > 0.0f)
			{
				sSoftRowAngular(p.mRoll[i], inPatch.mRollAxis[i], inInvI1, inInvI2);
				sSoftRowRigid(p.mRoll[i], 0.0f);
			}
		}
	}

	// The face grid's contact set, for the next step's first pass
	for (int i = 0; i < 6; ++i)
		ioTail.mActiveSet[i] = inPatch.mActiveSet[i];
	ioTail.mGridLong = inPatch.mGridLong;
	ioTail.mGridShort = inPatch.mGridShort;
}

bool ContactConstraintManager::HasSoftPatches(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd) const
{
	if (mSoftPatches == nullptr)
		return false;
	for (const uint32 *idx = inConstraintIdxBegin; idx < inConstraintIdxEnd; ++idx)
		if (mConstraints[*idx].mSoftPatchIndex != cNoSoftPatch)
			return true;
	return false;
}

bool ContactConstraintManager::FinalizeSoftPatches(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd)
{
	if (!HasSoftPatches(inConstraintIdxBegin, inConstraintIdxEnd))
		return false;

	// The pair-sum bound (SOFT-CONTACT brief 5.4): S_b sums k w over the compliant rows of each dynamic body;
	// static and kinematic bodies take 0, so the sums never depend on which neighbours share the island.
	// Each row's a = h^2 max(k w, S_body1 + S_body2), with w the row's free-body inverse effective mass.
	const Body *sum_body[cSoftMaxSumBodies];
	float sum_value[cSoftMaxSumBodies];
	int num_sums = 0;
	auto sum_index = [&](const Body *inBody) -> int
	{
		if (!inBody->IsDynamic())
			return -1;
		for (int i = 0; i < num_sums; ++i)
			if (sum_body[i] == inBody)
				return i;
		if (num_sums >= cSoftMaxSumBodies)
			return -2;
		sum_body[num_sums] = inBody;
		sum_value[num_sums] = 0.0f;
		return num_sums++;
	};

	for (const uint32 *idx = inConstraintIdxBegin; idx < inConstraintIdxEnd; ++idx)
	{
		const ContactConstraint &constraint = mConstraints[*idx];
		if (constraint.mSoftPatchIndex == cNoSoftPatch)
			continue;
		SoftPatchConstraint &p = mSoftPatches[constraint.mSoftPatchIndex];
		int i1 = sum_index(constraint.mBody1);
		int i2 = sum_index(constraint.mBody2);
		SoftRow *rows[6] = { &p.mNormal, &p.mTilt[0], &p.mTilt[1], &p.mFriction[0], &p.mFriction[1], &p.mFriction[2] };
		for (SoftRow *r : rows)
			if (r->mMode == SoftRowCompliant)
			{
				float kw = r->mStiffness * r->mInvEffectiveMass;
				if (i1 >= 0)
					sum_value[i1] += kw;
				if (i2 >= 0)
					sum_value[i2] += kw;
			}
	}

	for (const uint32 *idx = inConstraintIdxBegin; idx < inConstraintIdxEnd; ++idx)
	{
		const ContactConstraint &constraint = mConstraints[*idx];
		if (constraint.mSoftPatchIndex == cNoSoftPatch)
			continue;
		SoftPatchConstraint &p = mSoftPatches[constraint.mSoftPatchIndex];
		int i1 = sum_index(constraint.mBody1);
		int i2 = sum_index(constraint.mBody2);
		float pair_sum = (i1 >= 0? sum_value[i1] : 0.0f) + (i2 >= 0? sum_value[i2] : 0.0f);
		float h = p.mDeltaTime;

		SoftRow *rows[6] = { &p.mNormal, &p.mTilt[0], &p.mTilt[1], &p.mFriction[0], &p.mFriction[1], &p.mFriction[2] };
		for (SoftRow *r : rows)
		{
			if (r->mMode != SoftRowCompliant)
				continue;
			float w = r->mInvEffectiveMass;
			float k = r->mStiffness;
			float a = Square(h) * max(k * w, pair_sum);
			r->mTheta = a > 0.5f? 1.0f - 0.5f / a : 0.0f;

			// Material damping on every compliant row, c = 2 zeta sqrt(k m) with m the row's free-body mass
			float c = 2.0f * r->mDampingRatio * sqrt(k / w);
			float velocity_coefficient = c + r->mTheta * k * h;
			if (SpringPart::CalculateSpringPropertiesWithPreload(h, w, r->mPreload, velocity_coefficient, cSoftFixedEpsilon, r->mSoftness, r->mBias, r->mEffectiveMass))
			{
				// Across a gap the implicit share acts only on the predicted overlap: impulse B (d0 + h v_closing)
				if (r == &p.mNormal && p.mGap)
					r->mBias = -p.mSquash / h;
			}
			else
			{
				r->mMode = SoftRowFixed;
				++mSoftStats.mFixedRows;
			}
			if (r->mTheta > 0.0f)
				++mSoftStats.mDampedRows;
		}

		// A friction block with one fixed row is applied as fixed impulses throughout
		bool fixed = p.mFriction[0].mMode == SoftRowFixed || p.mFriction[1].mMode == SoftRowFixed || p.mFriction[2].mMode == SoftRowFixed;
		if (fixed)
		{
			p.mFrictionFixed = true;
			for (SoftRow &r : p.mFriction)
				if (r.mMode != SoftRowInactive)
					r.mMode = SoftRowFixed;
		}

		// The friction block's metric for the ellipse: J M^-1 J^T plus each row's softness
		if (p.mFriction[0].mMode != SoftRowInactive)
		{
			SoftPatchBodies b = sSoftPatchBodies(*constraint.mBody1, constraint.mBody1->IsDynamic()? constraint.mInvMass1 : 0.0f, *constraint.mBody2, constraint.mBody2->IsDynamic()? constraint.mInvMass2 : 0.0f);
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					p.mFrictionMatrix[3 * i + j] = sSoftRowCoupling(p.mFriction[i], p.mFriction[j], b) + (i == j? double(p.mFriction[i].mSoftness) : 0.0);
		}
	}

	return true;
}

void ContactConstraintManager::WarmStartSoftPatch(ContactConstraint &ioConstraint, float inWarmStartImpulseRatio)
{
	SoftPatchConstraint &p = mSoftPatches[ioConstraint.mSoftPatchIndex];
	Body &body1 = *ioConstraint.mBody1;
	Body &body2 = *ioConstraint.mBody2;
	SoftPatchBodies b = sSoftPatchBodies(body1, body1.IsDynamic()? ioConstraint.mInvMass1 : 0.0f, body2, body2.IsDynamic()? ioConstraint.mInvMass2 : 0.0f);
	const SoftContactTail &tail = *p.mTail;
	float h = p.mDeltaTime;
	float ratio = p.mWarm? inWarmStartImpulseRatio : 0.0f;

	Quat rotation2 = body2.GetRotation();
	Vec3 friction = rotation2 * Vec3::sLoadFloat3Unsafe(tail.mFrictionImpulse);
	Vec3 moment = rotation2 * Vec3::sLoadFloat3Unsafe(tail.mMoment);

	auto start = [&](SoftRow &ioRow, float inLast)
	{
		switch (ioRow.mMode)
		{
		case SoftRowCompliant:
		case SoftRowRigid:
			ioRow.mTotalLambda = inLast * ratio;
			break;
		case SoftRowFixed:
			ioRow.mTotalLambda = h * ioRow.mPreload;
			break;
		default:
			ioRow.mTotalLambda = 0.0f;
			break;
		}
	};

	// Start from last step's impulses (the lead's block_delta: the cluster correction follows)
	start(p.mNormal, p.mImpact? 0.0f : tail.mNormalImpulse);
	if (p.mNormal.mMode == SoftRowFixed)
		p.mNormal.mTotalLambda = max(p.mNormal.mTotalLambda, 0.0f);
	start(p.mCore, tail.mCoreImpulse);
	start(p.mFriction[0], p.mImpact? 0.0f : friction.Dot(p.mTangent1));
	start(p.mFriction[1], p.mImpact? 0.0f : friction.Dot(p.mTangent2));
	start(p.mFriction[2], p.mImpact? 0.0f : moment.Dot(p.mNormalAxis));
	for (int i = 0; i < 2; ++i)
	{
		start(p.mTilt[i], moment.Dot(p.mTilt[i].mAngular2));
		start(p.mRoll[i], moment.Dot(p.mRoll[i].mAngular2));
	}

	// A fixed friction block keeps within the soft-finger ellipse of the normal impulse it starts with. Past the static
	// ellipse it slides on the kinetic one, as a solved block does: its pre-slip deflection then follows the impulse it
	// carried (StoreSoftPatch), never the relative motion, or a spring clamped while the normal force dips would store
	// a deflection it could not carry and give it back when the force returns.
	if (p.mFrictionFixed)
	{
		double l[3] = { p.mFriction[0].mTotalLambda, p.mFriction[1].mTotalLambda, p.mFriction[2].mTotalLambda };
		double identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
		double L = double(max(p.mNormal.mTotalLambda, 0.0f));
		double probe[3] = { l[0], l[1], l[2] };
		if (sSoftFingerProject(identity, probe, double(p.mStaticFriction) * L, p.mSpinRadius))
		{
			sSoftFingerProject(identity, l, double(p.mSlidingFriction) * L, p.mSpinRadius);
			p.mSliding = true;
		}
		for (int i = 0; i < 3; ++i)
			p.mFriction[i].mTotalLambda = float(l[i]);
	}

	sSoftRowApply(p.mNormal, b, p.mNormal.mTotalLambda);
	sSoftRowApply(p.mCore, b, p.mCore.mTotalLambda);
	for (SoftRow &r : p.mFriction)
		sSoftRowApply(r, b, r.mTotalLambda);
	for (SoftRow &r : p.mTilt)
		sSoftRowApply(r, b, r.mTotalLambda);
	for (SoftRow &r : p.mRoll)
		sSoftRowApply(r, b, r.mTotalLambda);
}

void ContactConstraintManager::WarmStartSoftClusters(const uint32 *inConstraintIdxBegin, const uint32 *inConstraintIdxEnd)
{
	if (mSoftPatches == nullptr || !mSoftBlockWarmStart)
		return;

	// The island's soft patches, in its (sorted) order
	uint32 patch_constraint[cSoftMaxClusterPatches];
	int num_patches = 0;
	bool over = false;
	for (const uint32 *idx = inConstraintIdxBegin; idx < inConstraintIdxEnd; ++idx)
		if (mConstraints[*idx].mSoftPatchIndex != cNoSoftPatch)
		{
			if (num_patches < cSoftMaxClusterPatches)
				patch_constraint[num_patches++] = *idx;
			else
				over = true;
		}
	if (num_patches == 0)
		return;

	// Clusters: patches linked through shared dynamic bodies (union find, the first patch the root)
	int parent[cSoftMaxClusterPatches];
	for (int i = 0; i < num_patches; ++i)
		parent[i] = i;
	auto find = [&](int inI)
	{
		while (parent[inI] != inI)
			inI = parent[inI] = parent[parent[inI]];
		return inI;
	};
	for (int i = 0; i < num_patches; ++i)
	{
		const ContactConstraint &ci = mConstraints[patch_constraint[i]];
		for (int j = 0; j < i; ++j)
		{
			const ContactConstraint &cj = mConstraints[patch_constraint[j]];
			bool shared = (ci.mBody1->IsDynamic() && (ci.mBody1 == cj.mBody1 || ci.mBody1 == cj.mBody2))
				|| (ci.mBody2->IsDynamic() && (ci.mBody2 == cj.mBody1 || ci.mBody2 == cj.mBody2));
			if (shared)
			{
				int ri = find(i), rj = find(j);
				if (ri != rj)
					parent[max(ri, rj)] = min(ri, rj);
			}
		}
	}

	// A row of the cluster's block: which patch, which row
	struct BlockRow
	{
		int					mPatch;
		SoftRow *			mRow;
	};

	auto solve_block = [&](const int *inPatches, int inNumPatches)
	{
		BlockRow rows[cSoftBlockRowCap];
		int num_rows = 0;
		for (int k = 0; k < inNumPatches; ++k)
		{
			SoftPatchConstraint &p = mSoftPatches[mConstraints[patch_constraint[inPatches[k]]].mSoftPatchIndex];
			SoftRow *candidates[6] = { &p.mNormal, &p.mTilt[0], &p.mTilt[1], &p.mFriction[0], &p.mFriction[1], &p.mFriction[2] };
			for (SoftRow *r : candidates)
				if (r->mMode == SoftRowCompliant && num_rows < cSoftBlockRowCap)
					rows[num_rows++] = { k, r };
		}
		if (num_rows == 0)
			return;

		// W = J M^-1 J^T of the free bodies plus the softness, r = J v + b + softness lambda
		double W[cSoftBlockRowCap * cSoftBlockRowCap];
		double rhs[cSoftBlockRowCap];
		double delta[cSoftBlockRowCap];
		SoftPatchBodies bodies[64];
		const Body *body1[64], *body2[64];
		for (int k = 0; k < inNumPatches && k < 64; ++k)
		{
			ContactConstraint &c = mConstraints[patch_constraint[inPatches[k]]];
			bodies[k] = sSoftPatchBodies(*c.mBody1, c.mBody1->IsDynamic()? c.mInvMass1 : 0.0f, *c.mBody2, c.mBody2->IsDynamic()? c.mInvMass2 : 0.0f);
			body1[k] = c.mBody1;
			body2[k] = c.mBody2;
		}
		for (int i = 0; i < num_rows; ++i)
		{
			const SoftRow &ri = *rows[i].mRow;
			const SoftPatchBodies &bi = bodies[rows[i].mPatch];
			rhs[i] = -(double(sSoftRowVelocity(ri, bi)) + double(ri.mBias) + double(ri.mSoftness) * double(ri.mTotalLambda));
			for (int j = 0; j < num_rows; ++j)
			{
				const SoftRow &rj = *rows[j].mRow;
				const SoftPatchBodies &bj = bodies[rows[j].mPatch];
				double w = 0.0;
				for (int si = 0; si < 2; ++si)
				{
					const Body *bod_i = si == 0? body1[rows[i].mPatch] : body2[rows[i].mPatch];
					if (!bod_i->IsDynamic())
						continue;
					for (int sj = 0; sj < 2; ++sj)
					{
						const Body *bod_j = sj == 0? body1[rows[j].mPatch] : body2[rows[j].mPatch];
						if (bod_i != bod_j)
							continue;
						double sign = (si == 0? -1.0 : 1.0) * (sj == 0? -1.0 : 1.0);
						Vec3 ang_i = si == 0? ri.mAngular1 : ri.mAngular2;
						Vec3 inv_i_ang_j = sj == 0? rj.mInvIAngular1 : rj.mInvIAngular2;
						float inv_mass_j = sj == 0? bj.mInvMass1 : bj.mInvMass2;
						w += sign * (double(inv_mass_j) * double(ri.mLinear.Dot(rj.mLinear)) + double(ang_i.Dot(inv_i_ang_j)));
					}
				}
				W[i * num_rows + j] = w + (i == j? double(ri.mSoftness) : 0.0);
			}
		}
		if (!sSoftSolveDense(W, rhs, num_rows, delta))
			return;

		// Clamp: normals at zero, tilt within its arms, friction into the soft-finger ellipse
		for (int k = 0; k < inNumPatches; ++k)
		{
			SoftPatchConstraint &p = mSoftPatches[mConstraints[patch_constraint[inPatches[k]]].mSoftPatchIndex];
			int i_normal = -1, i_tilt[2] = { -1, -1 }, i_friction[3] = { -1, -1, -1 };
			for (int i = 0; i < num_rows; ++i)
				if (rows[i].mPatch == k)
				{
					if (rows[i].mRow == &p.mNormal)
						i_normal = i;
					for (int t = 0; t < 2; ++t)
						if (rows[i].mRow == &p.mTilt[t])
							i_tilt[t] = i;
					for (int f = 0; f < 3; ++f)
						if (rows[i].mRow == &p.mFriction[f])
							i_friction[f] = i;
				}
			float normal = p.mNormal.mTotalLambda;
			if (i_normal >= 0)
			{
				if (double(normal) + delta[i_normal] < 0.0)
					delta[i_normal] = -double(normal);
				normal = float(double(normal) + delta[i_normal]);
			}
			float L = max(normal, 0.0f);
			for (int t = 0; t < 2; ++t)
				if (i_tilt[t] >= 0)
				{
					double total = Clamp(double(p.mTilt[t].mTotalLambda) + delta[i_tilt[t]], -double(L) * p.mTiltArmNeg[t], double(L) * p.mTiltArmPos[t]);
					delta[i_tilt[t]] = total - double(p.mTilt[t].mTotalLambda);
				}
			if (i_friction[0] >= 0 && i_friction[1] >= 0)
			{
				double l[3];
				for (int f = 0; f < 3; ++f)
					l[f] = double(p.mFriction[f].mTotalLambda) + (i_friction[f] >= 0? delta[i_friction[f]] : 0.0);
				sSoftFingerProject(p.mFrictionMatrix, l, double(p.mStaticFriction) * double(L), p.mSpinRadius);
				for (int f = 0; f < 3; ++f)
					if (i_friction[f] >= 0)
						delta[i_friction[f]] = l[f] - double(p.mFriction[f].mTotalLambda);
			}
		}

		// Apply
		for (int i = 0; i < num_rows; ++i)
		{
			SoftRow &r = *rows[i].mRow;
			float d = float(delta[i]);
			r.mTotalLambda += d;
			sSoftRowApply(r, bodies[rows[i].mPatch], d);
		}
	};

	for (int root = 0; root < num_patches; ++root)
	{
		if (find(root) != root)
			continue;
		int members[cSoftMaxClusterPatches];
		int num_members = 0;
		int num_rows = 0;
		for (int i = root; i < num_patches; ++i)
			if (find(i) == root)
			{
				members[num_members++] = i;
				const SoftPatchConstraint &p = mSoftPatches[mConstraints[patch_constraint[i]].mSoftPatchIndex];
				const SoftRow *candidates[6] = { &p.mNormal, &p.mTilt[0], &p.mTilt[1], &p.mFriction[0], &p.mFriction[1], &p.mFriction[2] };
				for (const SoftRow *r : candidates)
					if (r->mMode == SoftRowCompliant)
						++num_rows;
			}
		if (num_rows <= cSoftBlockRowCap && num_members <= 64)
			solve_block(members, num_members);
		else
		{
			++mSoftStats.mClusterOverCap;
			for (int m = 0; m < num_members; ++m)
				solve_block(&members[m], 1);
		}
	}
	if (over)
		++mSoftStats.mClusterOverCap;
}

bool ContactConstraintManager::SolveSoftPatch(ContactConstraint &ioConstraint)
{
	SoftPatchConstraint &p = mSoftPatches[ioConstraint.mSoftPatchIndex];
	Body &body1 = *ioConstraint.mBody1;
	Body &body2 = *ioConstraint.mBody2;
	SoftPatchBodies b = sSoftPatchBodies(body1, body1.IsDynamic()? ioConstraint.mInvMass1 : 0.0f, body2, body2.IsDynamic()? ioConstraint.mInvMass2 : 0.0f);
	bool any = false;

	// The normal impulse limits friction, tilt and rolling resistance; it is solved last, as the rigid contact
	// solves its normals last, so these take the previous sweep's
	float L = max(p.mNormal.mTotalLambda, 0.0f);

	// The friction block, limited together by the soft-finger ellipse
	if (p.mFriction[0].mMode != SoftRowInactive && !p.mFrictionFixed)
	{
		double r[3], lambda[3], x[3];
		for (int i = 0; i < 3; ++i)
		{
			const SoftRow &row = p.mFriction[i];
			r[i] = -(double(sSoftRowVelocity(row, b)) + double(row.mBias) + double(row.mSoftness) * double(row.mTotalLambda));
			lambda[i] = row.mTotalLambda;
		}
		double K[9];
		for (int i = 0; i < 9; ++i)
			K[i] = p.mFrictionMatrix[i];
		if (sSoftSolveDense(K, r, 3, x))
		{
			double l[3] = { lambda[0] + x[0], lambda[1] + x[1], lambda[2] + x[2] };
			bool slides = false;
			double q_static = 0.0;
			{
				double mu_l = double(p.mStaticFriction) * double(L);
				double rs = max(double(p.mSpinRadius), cSoftMinSpinRadius);
				if (mu_l > 0.0)
					q_static = (l[0] * l[0] + l[1] * l[1]) / (mu_l * mu_l) + (l[2] * l[2]) / ((mu_l * rs) * (mu_l * rs));
				else
					q_static = (l[0] != 0.0 || l[1] != 0.0 || l[2] != 0.0)? 2.0 : 0.0;
			}
			if (q_static > 1.0)
			{
				// Sliding: the kinetic coefficient's ellipse
				sSoftFingerProject(p.mFrictionMatrix, l, double(p.mSlidingFriction) * double(L), p.mSpinRadius);
				slides = true;
			}
			p.mSliding = slides;
			for (int i = 0; i < 3; ++i)
			{
				float d = float(l[i] - lambda[i]);
				if (d != 0.0f)
				{
					p.mFriction[i].mTotalLambda += d;
					sSoftRowApply(p.mFriction[i], b, d);
					any = true;
				}
			}
		}
	}

	// Rolling resistance: a saturating damper
	for (int i = 0; i < p.mNumRoll; ++i)
		if (p.mRoll[i].mMode == SoftRowRigid)
			any |= sSoftRowSolve(p.mRoll[i], b, -L * p.mRollArm[i], L * p.mRollArm[i]) != 0.0f;

	// Tilt: the moment about each axis within the normal impulse times the distance to the patch edge on that side
	for (int i = 0; i < p.mNumTilt; ++i)
		if (p.mTilt[i].mMode == SoftRowCompliant)
			any |= sSoftRowSolve(p.mTilt[i], b, -L * p.mTiltArmNeg[i], L * p.mTiltArmPos[i]) != 0.0f;

	// The normal row and the core guard: a contact never pulls
	if (p.mNormal.mMode == SoftRowCompliant || p.mNormal.mMode == SoftRowRigid)
		any |= sSoftRowSolve(p.mNormal, b, 0.0f, FLT_MAX) != 0.0f;
	if (p.mCore.mMode == SoftRowRigid)
		any |= sSoftRowSolve(p.mCore, b, 0.0f, FLT_MAX) != 0.0f;

	return any;
}

void ContactConstraintManager::StoreSoftPatch(const ContactConstraint &inConstraint) const
{
	SoftPatchConstraint &p = mSoftPatches[inConstraint.mSoftPatchIndex];
	Body &body1 = *inConstraint.mBody1;
	Body &body2 = *inConstraint.mBody2;
	SoftPatchBodies b = sSoftPatchBodies(body1, body1.IsDynamic()? inConstraint.mInvMass1 : 0.0f, body2, body2.IsDynamic()? inConstraint.mInvMass2 : 0.0f);
	SoftContactTail &tail = *p.mTail;
	float h = p.mDeltaTime;
	Vec3 n = p.mNormalAxis;

	float lambda_normal = max(p.mNormal.mTotalLambda, 0.0f);
	float lambda_core = max(p.mCore.mTotalLambda, 0.0f);
	Vec3 friction = p.mFriction[0].mTotalLambda * p.mTangent1 + p.mFriction[1].mTotalLambda * p.mTangent2;
	Vec3 moment = p.mFriction[2].mTotalLambda * n;
	for (int i = 0; i < 2; ++i)
		moment += p.mTilt[i].mTotalLambda * p.mTilt[i].mAngular2 + p.mRoll[i].mTotalLambda * p.mRoll[i].mAngular2;

	// The tail, in body 2's frame. An impact's impulses do not warm start the spring that follows it.
	Quat to_body2 = body2.GetRotation().Conjugated();
	tail.mNormalImpulse = p.mImpact? 0.0f : lambda_normal;
	tail.mCoreImpulse = lambda_core;
	(to_body2 * (p.mImpact? Vec3::sZero() : friction)).StoreFloat3(&tail.mFrictionImpulse);
	(to_body2 * (p.mImpact? Vec3::sZero() : moment)).StoreFloat3(&tail.mMoment);

	// The pre-slip deflection: carried with the relative motion while sticking, set to the deflection that carries
	// the applied friction while sliding, and relaxed by rolling over the relaxation length
	Vec3 r1 = Vec3(p.mCentre - body1.GetCenterOfMassPosition());
	Vec3 r2 = Vec3(p.mCentre - body2.GetCenterOfMassPosition());
	Vec3 v1 = b.mRead1 != nullptr? b.mRead1->GetPointVelocityCOM(r1) : Vec3::sZero();
	Vec3 v2 = b.mRead2 != nullptr? b.mRead2->GetPointVelocityCOM(r2) : Vec3::sZero();
	Vec3 w1 = b.mRead1 != nullptr? b.mRead1->GetAngularVelocity() : Vec3::sZero();
	Vec3 w2 = b.mRead2 != nullptr? b.mRead2->GetAngularVelocity() : Vec3::sZero();
	Vec3 v_rel = v2 - v1;
	Vec3 w_rel = w2 - w1;
	float shear1 = 0.0f, shear2 = 0.0f, spin = 0.0f;
	if (!p.mRigidStick && !p.mImpact && p.mFriction[0].mMode != SoftRowInactive)
	{
		if (p.mSliding && p.mShearStiffness > 0.0f)
		{
			shear1 = -p.mFriction[0].mTotalLambda / (h * p.mShearStiffness);
			shear2 = -p.mFriction[1].mTotalLambda / (h * p.mShearStiffness);
			spin = p.mSpinStiffness > 0.0f? -p.mFriction[2].mTotalLambda / (h * p.mSpinStiffness) : 0.0f;
		}
		else
		{
			shear1 = p.mShear[0] + h * v_rel.Dot(p.mTangent1);
			shear2 = p.mShear[1] + h * v_rel.Dot(p.mTangent2);
			spin = p.mSpin + h * w_rel.Dot(n);
		}
		if (p.mRelaxationLength > 0.0f)
		{
			float rolling_speed = (w_rel - w_rel.Dot(n) * n).Length() * p.mRollingRadius;
			float relax = std::exp(-rolling_speed * h / p.mRelaxationLength);
			shear1 *= relax;
			shear2 *= relax;
			spin *= relax;
		}
		else if (p.mRollingRadius > 0.0f)
			shear1 = shear2 = spin = 0.0f;
	}
	(to_body2 * (shear1 * p.mTangent1 + shear2 * p.mTangent2)).StoreFloat3(&tail.mShear);
	tail.mSpinDeflection = spin;

	// The overlap the solve leaves for the end of the step, at the normal row
	tail.mEndSquash = p.mSquash - h * sSoftRowVelocity(p.mNormal.mMode != SoftRowInactive? p.mNormal : p.mCore, b);
	tail.mFlags |= SoftContactTail::Solved;

	// The solved-contact report, built from the rows
	if (inConstraint.mReportSolved && mContactListener != nullptr)
	{
		RVec3 deep = body1.GetCenterOfMassTransform() * Vec3::sLoadFloat3Unsafe(p.mDeepLocal1);
		deep = 0.5_r * (deep + body2.GetCenterOfMassTransform() * Vec3::sLoadFloat3Unsafe(p.mDeepLocal2));
		RVec3 normal_point = p.mImpact? deep : p.mCentre;
		float pressing = lambda_normal + lambda_core;
		RVec3 centre = pressing > 0.0f? (Real(lambda_normal) * normal_point + Real(lambda_core) * deep) / Real(pressing) : p.mCentre;

		ContactSolvedReport report;
		report.mWorldNormal = n;
		report.mCentre = centre;
		report.mLinearImpulse = pressing * n + friction;
		report.mAngularImpulse = moment
			+ Vec3(normal_point - centre).Cross(lambda_normal * n)
			+ Vec3(deep - centre).Cross(lambda_core * n)
			+ Vec3(p.mCentre - centre).Cross(friction);
		report.mNormalImpulse = pressing;
		report.mArea = p.mArea;
		if (p.mSliding)
			report.mFrictionUse = 1.0f;
		else
		{
			float mu_n = p.mStaticFriction * lambda_normal;
			float use = 0.0f;
			if (mu_n > 0.0f)
			{
				use = Square(friction.Length() / mu_n);
				float rs = p.mSpinRadius;
				if (rs > 0.0f)
					use += Square(p.mFriction[2].mTotalLambda / (mu_n * rs));
				else if (p.mFriction[2].mTotalLambda != 0.0f)
					use = 1.0f;
				use = min(sqrt(use), 1.0f);
			}
			report.mFrictionUse = use;
		}

		SubShapeIDPair pair(body1.GetID(), inConstraint.mSubShapeID1, body2.GetID(), inConstraint.mSubShapeID2);
		mContactListener->OnContactSolved(body1, body2, pair, report);
	}
}

bool ContactConstraintManager::SolveSoftPatchPosition(ContactConstraint &ioConstraint)
{
	SoftPatchConstraint &p = mSoftPatches[ioConstraint.mSoftPatchIndex];
	Body &body1 = *ioConstraint.mBody1;
	Body &body2 = *ioConstraint.mBody2;
	Vec3 n = p.mNormalAxis;

	// The deepest point on each surface where the bodies are now
	RVec3 p1 = body1.GetCenterOfMassTransform() * Vec3::sLoadFloat3Unsafe(p.mDeepLocal1);
	RVec3 p2 = body2.GetCenterOfMassTransform() * Vec3::sLoadFloat3Unsafe(p.mDeepLocal2);
	float surface_separation = Vec3(p2 - p1).Dot(n);

	// An impact's overlap goes to the position pass like a rigid penetration; otherwise only an approach past
	// the core (the layer squashed through) is corrected
	float separation = p.mImpact? surface_separation : surface_separation + p.mLayer;
	separation = max(separation + mPhysicsSettings.mPenetrationSlop, -mPhysicsSettings.mMaxPenetrationDistance);
	if (separation >= 0.0f)
		return false;

	RVec3 mid = 0.5_r * (p1 + p2);
	Vec3 r1 = Vec3(mid - body1.GetCenterOfMassPosition());
	Vec3 r2 = Vec3(mid - body2.GetCenterOfMassPosition());
	p.mPositionPart.CalculateConstraintPropertiesWithMassOverride(body1, ioConstraint.mInvMass1, ioConstraint.mInvInertiaScale1, r1, body2, ioConstraint.mInvMass2, ioConstraint.mInvInertiaScale2, r2, n);
	return p.mPositionPart.SolvePositionConstraintWithMassOverride(body1, ioConstraint.mInvMass1, body2, ioConstraint.mInvMass2, n, separation, mPhysicsSettings.mBaumgarte);
}

JPH_NAMESPACE_END
