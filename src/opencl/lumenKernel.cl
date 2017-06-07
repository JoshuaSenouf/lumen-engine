__constant float PI = 3.14159265359f;
__constant float FOV_ANGLE = 0.5135f;
__constant float EPSILON = 0.00003f;


enum materialType
{
	DIFFUSE,
	SPECULAR,
	REFRACT
};


typedef struct RayObject
{
	float3 origin;
	float3 direction;
} RayObject;


typedef struct SphereObject
{
	float radius;
	float3 position;
	float3 color;
	float3 emissiveColor;
	enum materialType material;
} SphereObject;


/* Random number function from Samuel Lapere */
static float getrandom(unsigned int *seed1, unsigned int *seed2)
{
	*seed1 = 36969 * ((*seed1) & 65535) + ((*seed1) >> 16);
	*seed2 = 18000 * ((*seed2) & 65535) + ((*seed2) >> 16);

	unsigned int ires = ((*seed1) << 16) + (*seed2);

	union
	{
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;

	return (res.f - 2.0f) / 2.0f;
}


/* Convert HDR floating point value to an SRGB integer that can be read by OpenGL and saved on disk */
inline int hdrToSGRB(float x)
{
	return pow(clamp(x, 0.0f, 1.0f), 1.0f / 2.2f) * 255;
}


RayObject getPrimaryRay(const int pixelX, const int pixelY, const int width, const int height) {

	float floatX = (float)pixelX / (float)width; /* clamp pixelX from an int to a saturated float */
	float floatY = (float)pixelY / (float)height; 

	float renderAspectRatio = (float)(width) / (float)(height);
	floatX = (floatX - 0.5f) * renderAspectRatio;
	floatY = floatY - 0.5f;

	float3 pixelPos = (float3)(floatX, -floatY, 0.0f);

	RayObject cameraRay;
	cameraRay.origin = (float3)(0.0f, 0.1f, 2.0f); /* Starting viewpoint */
	cameraRay.direction = normalize(pixelPos - cameraRay.origin); /* Camera/Pixel vector */

	return cameraRay;
}


float checkSphereIntersect(const SphereObject* sphere, const RayObject* ray)
{
	float3 op = sphere->position - ray->origin;
	float t, eps = 1e-4;
	float b = dot(op, ray->direction);

	float quadDis = (b * b) - dot(op, op) + (sphere->radius * sphere->radius);

	if (quadDis < 0)
		return 0;
	else
		quadDis = sqrt(quadDis);

	return (t = b - quadDis) >  eps ? t : ((t = b + quadDis) > eps ? t : 0);
}


bool checkSceneIntersect(const RayObject* ray, __constant SphereObject* spheresList, float* closestSphereDist, int* closestSphereID, const int sphereCount)
{
	float distance;
	float inf = 1e20;
	*closestSphereDist = inf;

	for (int i = 0; i < sphereCount; i++)
	{
		SphereObject sphere = spheresList[i];

		if ((distance = checkSphereIntersect(&sphere, ray)) && distance < *closestSphereDist)
		{
			*closestSphereDist = distance;
			*closestSphereID = i;
		}
	}

	return *closestSphereDist < inf;
}


float3 computeCosineWeightedImportanceSampling(float3 localW, float3 localU, float3 localV, float rand1, float rand2, float sqrtRand2)
{
	return normalize(localU * cos(rand1) * sqrtRand2 + localV * sin(rand1) * sqrtRand2 + localW * sqrt(1 - rand2));
}


float3 computePerfectlyReflectedRay(float3 rayDirection, float3 intersectionNormal)
{
	return rayDirection - 2.0f * intersectionNormal * dot(intersectionNormal, rayDirection);
}


float3 computeRadiance(const RayObject* ray, __constant SphereObject* spheresList, const int lightBounces, const int sphereCount, unsigned int *seed1, unsigned int *seed2)
{
	RayObject tempRay = *ray;

	float3 colorAccumulation = (float3)(0.0f, 0.0f, 0.0f);
	float3 colorMask = (float3)(1.0f, 1.0f, 1.0f);

	for (int bounces = 0; bounces < lightBounces; bounces++)
	{
		float closestSphereDist;
		int closestSphereID = 0;

		if (!checkSceneIntersect(&tempRay, spheresList, &closestSphereDist, &closestSphereID, sphereCount))
			return colorAccumulation += colorMask * (float3)(0.15f, 0.15f, 0.25f);

		const SphereObject hitSphere = spheresList[closestSphereID];
		float3 hitCoord = tempRay.origin + tempRay.direction * closestSphereDist;
		float3 hitNormal = normalize(hitCoord - hitSphere.position);
		float3 hitFrontNormal = dot(hitNormal, tempRay.direction) < 0.0f ? hitNormal : hitNormal * -1.0f;

		colorAccumulation += colorMask * hitSphere.emissiveColor;

		float random1 = 2.0f * PI * getrandom(seed1, seed2);
		float random2 = getrandom(seed1, seed2);
		float random2Square = sqrt(random2);

		float3 nextRayDir;

		if (hitSphere.material == 1)
		{
			float3 localOrthoW = hitFrontNormal;
			float3 localOrthoU = normalize(cross((fabs(localOrthoW.x) > 0.1f ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f)), localOrthoW));
			float3 localOrthoV = cross(localOrthoW, localOrthoU);

			nextRayDir = computeCosineWeightedImportanceSampling(localOrthoW, localOrthoU, localOrthoV, random1, random2, random2Square);
			hitCoord += hitFrontNormal * EPSILON;

			colorMask *= dot(nextRayDir, hitFrontNormal);
		}

		else if (hitSphere.material == 2)
		{
			nextRayDir = computePerfectlyReflectedRay(tempRay.direction, hitNormal);
			hitCoord += hitFrontNormal * EPSILON;
		}

		else if (hitSphere.material == 3)
		{

		}

		tempRay.direction = nextRayDir;
		tempRay.origin = hitCoord;

		colorMask *= hitSphere.color;
		colorMask *= 1.5f;
	}

	return colorAccumulation;
}


__kernel void lumenRender(__constant SphereObject* spheresList, const int renderWidth, const int renderHeight, const int samples, const int lightBounces, const int sphereCount, __global float3* renderOutput)
{
	unsigned int itemID = get_global_id(0); /* Global ID of the work item for the pixel we are working on */
	unsigned int pixelX = itemID % renderWidth;
	unsigned int pixelY = itemID / renderWidth;

	unsigned int seed1 = pixelX;
	unsigned int seed2 = pixelY;

	RayObject cameraRay = getPrimaryRay(pixelX, pixelY, renderWidth, renderHeight);

	float3 tempColor = (float3)(0.0f, 0.0f, 0.0f);

	for (int sample = 0; sample < samples; sample++)
	{
		tempColor += computeRadiance(&cameraRay, spheresList, lightBounces, sphereCount, &seed1, &seed2) * (1.0f / samples);
	}

	float3 pixelColor = (float3)(hdrToSGRB(tempColor.x), hdrToSGRB(tempColor.y), hdrToSGRB(tempColor.z));

	renderOutput[itemID] = pixelColor;
}
