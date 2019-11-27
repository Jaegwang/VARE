
#ifndef _BoraZalesakSphere_h_
#define _BoraZalesakSphere_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

static const Vector3<float> ZalesakSphereVertices[369] =
{
	{ -3.02145, 15.36, 1.59175e-06 }, 
	{ -3.05979, 15.3692, 1.59175e-06 }, 
	{ -3.02145, 15.3692, -0.242078 }, 
	{ -3.02145, 15.3692, 0.242081 }, 
	{ -4.49528, 15.9638, 1.59175e-06 }, 
	{ -4.27526, 15.9638, -1.38912 }, 
	{ -3.02145, 15.4177, -0.981728 }, 
	{ -3.02145, 15.4177, 0.981731 }, 
	{ -4.27526, 15.9638, 1.38912 }, 
	{ -5.82008, 16.7757, 1.59175e-06 }, 
	{ -5.53522, 16.7757, -1.7985 }, 
	{ -3.63676, 15.9638, -2.64226 }, 
	{ -3.02145, 15.6488, -2.19521 }, 
	{ -3.02145, 15.6488, 2.19522 }, 
	{ -3.63676, 15.9638, 2.64226 }, 
	{ -5.53522, 16.7757, 1.7985 }, 
	{ -4.70854, 16.7757, -3.42095 }, 
	{ -7.00156, 17.7847, 1.59175e-06 }, 
	{ -6.65888, 17.7847, -2.1636 }, 
	{ -3.02145, 15.9638, -3.25756 }, 
	{ -3.02145, 15.9638, 3.25757 }, 
	{ -4.70854, 16.7757, 3.42096 }, 
	{ -6.65888, 17.7847, 2.1636 }, 
	{ -3.42095, 16.7757, -4.70854 }, 
	{ -3.02145, 16.3591, -4.15867 }, 
	{ -5.66438, 17.7847, -4.11541 }, 
	{ -8.01065, 18.9662, 1.59175e-06 }, 
	{ -7.61858, 18.9662, -2.47542 }, 
	{ -3.02145, 16.3591, 4.15868 }, 
	{ -3.42095, 16.7757, 4.70854 }, 
	{ -5.66438, 17.7847, 4.11542 }, 
	{ -7.61858, 18.9662, 2.47543 }, 
	{ -4.11542, 17.7847, -5.66438 }, 
	{ -3.02145, 16.7757, -4.9121 }, 
	{ -6.48075, 18.9662, -4.70854 }, 
	{ -8.82248, 20.291, 1.59175e-06 }, 
	{ -8.39068, 20.291, -2.7263 }, 
	{ -3.02145, 16.7757, 4.9121 }, 
	{ -4.11541, 17.7847, 5.66438 }, 
	{ -6.48075, 18.9662, 4.70854 }, 
	{ -8.39068, 20.291, 2.7263 }, 
	{ -3.02145, 17.7847, -6.22178 }, 
	{ -4.70854, 18.9662, -6.48075 }, 
	{ -7.13754, 20.291, -5.18572 }, 
	{ -9.41708, 21.7265, 1.59175e-06 }, 
	{ -8.95618, 21.7265, -2.91004 }, 
	{ -3.02145, 17.7847, 6.22179 }, 
	{ -4.70854, 18.9662, 6.48075 }, 
	{ -7.13754, 20.291, 5.18573 }, 
	{ -8.95618, 21.7265, 2.91004 }, 
	{ -3.02145, 18.9662, -7.34036 }, 
	{ -5.18573, 20.291, -7.13754 }, 
	{ -7.61858, 21.7265, -5.53522 }, 
	{ -9.7798, 23.2373, 1.59175e-06 }, 
	{ -9.30114, 23.2373, -3.02212 }, 
	{ -3.02145, 18.9662, 7.34036 }, 
	{ -5.18572, 20.291, 7.13754 }, 
	{ -7.61858, 21.7265, 5.53522 }, 
	{ -9.30114, 23.2373, 3.02213 }, 
	{ -3.02145, 20.291, -8.24029 }, 
	{ -5.53522, 21.7265, -7.61858 }, 
	{ -7.91202, 23.2373, -5.74842 }, 
	{ -9.9017, 24.7863, 1.59175e-06 }, 
	{ -9.41708, 24.7863, -3.05979 }, 
	{ -3.02145, 20.291, 8.24029 }, 
	{ -5.53522, 21.7265, 7.61858 }, 
	{ -7.91202, 23.2373, 5.74842 }, 
	{ -9.41708, 24.7863, 3.0598 }, 
	{ -3.02145, 21.7265, -8.89941 }, 
	{ -5.74842, 23.2373, -7.91202 }, 
	{ -8.01065, 24.7863, -5.82008 }, 
	{ -9.7798, 26.3353, 1.59175e-06 }, 
	{ -9.30114, 26.3353, -3.02212 }, 
	{ -3.02145, 21.7265, 8.89941 }, 
	{ -5.74842, 23.2373, 7.91202 }, 
	{ -8.01065, 24.7863, 5.82008 }, 
	{ -9.30114, 26.3353, 3.02213 }, 
	{ -3.02212, 23.2373, -9.30114 }, 
	{ -3.02145, 23.2283, -9.29908 }, 
	{ -5.82008, 24.7863, -8.01065 }, 
	{ -7.91202, 26.3353, -5.74842 }, 
	{ -9.41708, 27.8461, 1.59175e-06 }, 
	{ -8.95618, 27.8461, -2.91004 }, 
	{ -3.02145, 23.2283, 9.29908 }, 
	{ -3.02212, 23.2373, 9.30114 }, 
	{ -5.82008, 24.7863, 8.01065 }, 
	{ -7.91202, 26.3353, 5.74842 }, 
	{ -8.95618, 27.8461, 2.91004 }, 
	{ -3.05979, 24.7863, -9.41708 }, 
	{ -3.02145, 23.2373, -9.30125 }, 
	{ -5.74842, 26.3353, -7.91202 }, 
	{ -7.61858, 27.8461, -5.53522 }, 
	{ -8.82248, 29.2816, 1.59175e-06 }, 
	{ -8.39068, 29.2816, -2.7263 }, 
	{ -3.02145, 23.2373, 9.30125 }, 
	{ -3.05979, 24.7863, 9.41708 }, 
	{ -5.74842, 26.3353, 7.91202 }, 
	{ -7.61858, 27.8461, 5.53522 }, 
	{ -8.39068, 29.2816, 2.7263 }, 
	{ -3.02145, 24.7863, -9.42315 }, 
	{ -3.02212, 26.3353, -9.30114 }, 
	{ -5.53522, 27.8461, -7.61858 }, 
	{ -7.13754, 29.2816, -5.18572 }, 
	{ -8.01065, 30.6064, 1.59175e-06 }, 
	{ -7.61858, 30.6064, -2.47542 }, 
	{ -3.02145, 24.7863, 9.42315 }, 
	{ -3.02212, 26.3353, 9.30114 }, 
	{ -5.53522, 27.8461, 7.61858 }, 
	{ -7.13754, 29.2816, 5.18573 }, 
	{ -7.61858, 30.6064, 2.47543 }, 
	{ -3.02145, 26.3353, -9.30125 }, 
	{ -2.91004, 27.8461, -8.95618 }, 
	{ -2.92605, 27.6303, -9.00546 }, 
	{ -3.02145, 27.6303, -8.95685 }, 
	{ -3.02145, 26.3443, -9.29908 }, 
	{ -5.18573, 29.2816, -7.13754 }, 
	{ -6.48075, 30.6064, -4.70854 }, 
	{ -7.00156, 31.7879, 1.59175e-06 }, 
	{ -6.65888, 31.7879, -2.1636 }, 
	{ -3.02145, 26.3353, 9.30125 }, 
	{ -3.02145, 26.3443, 9.29908 }, 
	{ -3.02145, 27.6303, 8.95685 }, 
	{ -2.92605, 27.6303, 9.00546 }, 
	{ -2.91004, 27.8461, 8.95618 }, 
	{ -5.18572, 29.2816, 7.13754 }, 
	{ -6.48075, 30.6064, 4.70854 }, 
	{ -6.65888, 31.7879, 2.1636 }, 
	{ -2.7263, 29.2816, -8.39068 }, 
	{ 1.06117e-06, 27.8461, -9.41708 }, 
	{ 1.06117e-06, 27.6303, -9.4689 }, 
	{ -4.70854, 30.6064, -6.48075 }, 
	{ -5.66438, 31.7879, -4.11541 }, 
	{ -5.82008, 32.7969, 1.59175e-06 }, 
	{ -5.53522, 32.7969, -1.7985 }, 
	{ 7.78971e-07, 27.6303, 9.4689 }, 
	{ 7.80515e-07, 27.8461, 9.41708 }, 
	{ -2.7263, 29.2816, 8.39068 }, 
	{ -4.70854, 30.6064, 6.48075 }, 
	{ -5.66438, 31.7879, 4.11542 }, 
	{ -5.53522, 32.7969, 1.7985 }, 
	{ 1.06117e-06, 29.2816, -8.82248 }, 
	{ -2.47543, 30.6064, -7.61858 }, 
	{ 2.91004, 27.8461, -8.95618 }, 
	{ 2.92605, 27.6303, -9.00546 }, 
	{ -4.11542, 31.7879, -5.66438 }, 
	{ -4.70854, 32.7969, -3.42095 }, 
	{ -4.49528, 33.6088, 1.59175e-06 }, 
	{ -4.27526, 33.6088, -1.38912 }, 
	{ 2.92605, 27.6303, 9.00546 }, 
	{ 2.91004, 27.8461, 8.95618 }, 
	{ 7.98235e-07, 29.2816, 8.82248 }, 
	{ -2.47543, 30.6064, 7.61858 }, 
	{ -4.11541, 31.7879, 5.66438 }, 
	{ -4.70854, 32.7969, 3.42096 }, 
	{ -4.27526, 33.6088, 1.38912 }, 
	{ 2.7263, 29.2816, -8.39068 }, 
	{ 1.06117e-06, 30.6064, -8.01065 }, 
	{ -2.1636, 31.7879, -6.65888 }, 
	{ 5.74843, 26.3353, -7.91202 }, 
	{ 3.02213, 26.3353, -9.30114 }, 
	{ 3.02145, 26.3443, -9.29908 }, 
	{ 3.02145, 27.6303, -8.95685 }, 
	{ 5.53522, 27.8461, -7.61858 }, 
	{ -3.42095, 32.7969, -4.70854 }, 
	{ -3.63676, 33.6088, -2.64226 }, 
	{ -3.05979, 34.2034, 1.59175e-06 }, 
	{ -2.91004, 34.2034, -0.945527 }, 
	{ 3.02212, 26.3353, 9.30114 }, 
	{ 5.74842, 26.3353, 7.91202 }, 
	{ 5.53522, 27.8461, 7.61858 }, 
	{ 3.02145, 27.6303, 8.95685 }, 
	{ 3.02145, 26.3443, 9.29908 }, 
	{ 2.7263, 29.2816, 8.39068 }, 
	{ 8.2243e-07, 30.6064, 8.01065 }, 
	{ -2.1636, 31.7879, 6.65888 }, 
	{ -3.42095, 32.7969, 4.70854 }, 
	{ -3.63676, 33.6088, 2.64226 }, 
	{ -2.91004, 34.2034, 0.94553 }, 
	{ 5.18573, 29.2816, -7.13754 }, 
	{ 2.47543, 30.6064, -7.61858 }, 
	{ 1.06117e-06, 31.7879, -7.00156 }, 
	{ -1.7985, 32.7969, -5.53522 }, 
	{ 7.91203, 26.3353, -5.74842 }, 
	{ 7.61858, 27.8461, -5.53522 }, 
	{ 5.82008, 24.7863, -8.01065 }, 
	{ 3.0598, 24.7863, -9.41708 }, 
	{ 3.02145, 26.3353, -9.30125 }, 
	{ -2.64226, 33.6088, -3.63676 }, 
	{ -2.47543, 34.2034, -1.7985 }, 
	{ -1.54897, 34.5661, 1.59175e-06 }, 
	{ -1.47316, 34.5661, -0.478656 }, 
	{ 3.02145, 26.3353, 9.30125 }, 
	{ 3.0598, 24.7863, 9.41708 }, 
	{ 5.82008, 24.7863, 8.01065 }, 
	{ 7.91202, 26.3353, 5.74842 }, 
	{ 7.61858, 27.8461, 5.53522 }, 
	{ 5.18573, 29.2816, 7.13754 }, 
	{ 2.47543, 30.6064, 7.61858 }, 
	{ 8.52503e-07, 31.7879, 7.00156 }, 
	{ -1.7985, 32.7969, 5.53522 }, 
	{ -2.64226, 33.6088, 3.63676 }, 
	{ -2.47543, 34.2034, 1.7985 }, 
	{ -1.47316, 34.5661, 0.478659 }, 
	{ 7.13754, 29.2816, -5.18573 }, 
	{ 4.70854, 30.6064, -6.48075 }, 
	{ 2.1636, 31.7879, -6.65888 }, 
	{ 1.06117e-06, 32.7969, -5.82008 }, 
	{ -1.38912, 33.6088, -4.27526 }, 
	{ 9.30115, 26.3353, -3.02212 }, 
	{ 8.95618, 27.8461, -2.91004 }, 
	{ 8.01065, 24.7863, -5.82008 }, 
	{ 5.74843, 23.2373, -7.91202 }, 
	{ 3.02213, 23.2373, -9.30114 }, 
	{ 3.02145, 24.7863, -9.42316 }, 
	{ -1.7985, 34.2034, -2.47543 }, 
	{ -1.25314, 34.5661, -0.910459 }, 
	{ 1.06117e-06, 34.688, 1.59175e-06 }, 
	{ 3.02145, 24.7863, 9.42315 }, 
	{ 3.02212, 23.2373, 9.30114 }, 
	{ 5.74842, 23.2373, 7.91202 }, 
	{ 8.01065, 24.7863, 5.82008 }, 
	{ 9.30114, 26.3353, 3.02213 }, 
	{ 8.95618, 27.8461, 2.91004 }, 
	{ 7.13754, 29.2816, 5.18573 }, 
	{ 4.70854, 30.6064, 6.48075 }, 
	{ 2.1636, 31.7879, 6.65888 }, 
	{ 8.87714e-07, 32.7969, 5.82008 }, 
	{ -1.38912, 33.6088, 4.27527 }, 
	{ -1.7985, 34.2034, 2.47543 }, 
	{ -1.25314, 34.5661, 0.910462 }, 
	{ 8.39069, 29.2816, -2.7263 }, 
	{ 6.48075, 30.6064, -4.70854 }, 
	{ 4.11542, 31.7879, -5.66438 }, 
	{ 1.7985, 32.7969, -5.53522 }, 
	{ 1.06117e-06, 33.6088, -4.49528 }, 
	{ -0.945528, 34.2034, -2.91004 }, 
	{ 9.7798, 26.3353, 1.59175e-06 }, 
	{ 9.41708, 27.8461, 1.59175e-06 }, 
	{ 9.41709, 24.7863, -3.05979 }, 
	{ 7.91203, 23.2373, -5.74842 }, 
	{ 3.02145, 23.2283, -9.29908 }, 
	{ 5.53522, 21.7265, -7.61858 }, 
	{ 3.02145, 21.7265, -8.89941 }, 
	{ 3.02145, 23.2373, -9.30125 }, 
	{ -0.91046, 34.5661, -1.25314 }, 
	{ 3.02145, 23.2373, 9.30125 }, 
	{ 3.02145, 21.7265, 8.89941 }, 
	{ 5.53522, 21.7265, 7.61858 }, 
	{ 3.02145, 23.2283, 9.29908 }, 
	{ 7.91202, 23.2373, 5.74842 }, 
	{ 9.41708, 24.7863, 3.0598 }, 
	{ 8.39068, 29.2816, 2.7263 }, 
	{ 6.48075, 30.6064, 4.70854 }, 
	{ 4.11542, 31.7879, 5.66438 }, 
	{ 1.7985, 32.7969, 5.53522 }, 
	{ 9.27196e-07, 33.6088, 4.49528 }, 
	{ -0.945528, 34.2034, 2.91004 }, 
	{ -0.910459, 34.5661, 1.25314 }, 
	{ 8.82248, 29.2816, 1.59175e-06 }, 
	{ 7.61858, 30.6064, -2.47543 }, 
	{ 5.66439, 31.7879, -4.11542 }, 
	{ 3.42096, 32.7969, -4.70854 }, 
	{ 1.38912, 33.6088, -4.27527 }, 
	{ 1.06117e-06, 34.2034, -3.05979 }, 
	{ -0.478657, 34.5661, -1.47315 }, 
	{ 9.9017, 24.7863, 1.59175e-06 }, 
	{ 9.30115, 23.2373, -3.02212 }, 
	{ 7.61858, 21.7265, -5.53522 }, 
	{ 5.18573, 20.291, -7.13754 }, 
	{ 3.02145, 20.291, -8.24029 }, 
	{ 3.02145, 20.291, 8.24029 }, 
	{ 5.18573, 20.291, 7.13754 }, 
	{ 7.61858, 21.7265, 5.53522 }, 
	{ 9.30114, 23.2373, 3.02213 }, 
	{ 7.61858, 30.6064, 2.47543 }, 
	{ 5.66438, 31.7879, 4.11542 }, 
	{ 3.42096, 32.7969, 4.70854 }, 
	{ 1.38912, 33.6088, 4.27527 }, 
	{ 9.69977e-07, 34.2034, 3.0598 }, 
	{ -0.478656, 34.5661, 1.47316 }, 
	{ 8.01065, 30.6064, 1.59175e-06 }, 
	{ 6.65889, 31.7879, -2.1636 }, 
	{ 4.70854, 32.7969, -3.42095 }, 
	{ 2.64226, 33.6088, -3.63676 }, 
	{ 0.94553, 34.2034, -2.91004 }, 
	{ 1.06117e-06, 34.5661, -1.54897 }, 
	{ 9.7798, 23.2373, 1.59175e-06 }, 
	{ 8.95618, 21.7265, -2.91004 }, 
	{ 7.13754, 20.291, -5.18573 }, 
	{ 4.70854, 18.9662, -6.48075 }, 
	{ 3.02145, 18.9662, -7.34036 }, 
	{ 3.02145, 18.9662, 7.34036 }, 
	{ 4.70854, 18.9662, 6.48075 }, 
	{ 7.13754, 20.291, 5.18573 }, 
	{ 8.95618, 21.7265, 2.91004 }, 
	{ 6.65888, 31.7879, 2.1636 }, 
	{ 4.70854, 32.7969, 3.42096 }, 
	{ 2.64226, 33.6088, 3.63676 }, 
	{ 0.94553, 34.2034, 2.91004 }, 
	{ 1.015e-06, 34.5661, 1.54897 }, 
	{ 7.00156, 31.7879, 1.59175e-06 }, 
	{ 5.53522, 32.7969, -1.7985 }, 
	{ 3.63676, 33.6088, -2.64226 }, 
	{ 1.7985, 34.2034, -2.47543 }, 
	{ 0.478659, 34.5661, -1.47316 }, 
	{ 9.41708, 21.7265, 1.59175e-06 }, 
	{ 8.39069, 20.291, -2.7263 }, 
	{ 6.48075, 18.9662, -4.70854 }, 
	{ 4.11542, 17.7847, -5.66438 }, 
	{ 3.02145, 17.7847, -6.22179 }, 
	{ 3.02145, 17.7847, 6.22178 }, 
	{ 4.11542, 17.7847, 5.66438 }, 
	{ 6.48075, 18.9662, 4.70854 }, 
	{ 8.39068, 20.291, 2.7263 }, 
	{ 5.53522, 32.7969, 1.7985 }, 
	{ 3.63676, 33.6088, 2.64226 }, 
	{ 1.7985, 34.2034, 2.47543 }, 
	{ 0.478658, 34.5661, 1.47316 }, 
	{ 5.82008, 32.7969, 1.59175e-06 }, 
	{ 4.27527, 33.6088, -1.38912 }, 
	{ 2.47543, 34.2034, -1.7985 }, 
	{ 0.910462, 34.5661, -1.25314 }, 
	{ 8.82248, 20.291, 1.59175e-06 }, 
	{ 7.61858, 18.9662, -2.47543 }, 
	{ 5.66439, 17.7847, -4.11542 }, 
	{ 3.42096, 16.7757, -4.70854 }, 
	{ 3.02145, 16.7757, -4.9121 }, 
	{ 3.02145, 16.7757, 4.9121 }, 
	{ 3.42096, 16.7757, 4.70854 }, 
	{ 5.66438, 17.7847, 4.11542 }, 
	{ 7.61858, 18.9662, 2.47543 }, 
	{ 4.27527, 33.6088, 1.38912 }, 
	{ 2.47543, 34.2034, 1.7985 }, 
	{ 0.910461, 34.5661, 1.25314 }, 
	{ 4.49528, 33.6088, 1.59175e-06 }, 
	{ 2.91004, 34.2034, -0.945527 }, 
	{ 1.25314, 34.5661, -0.910459 }, 
	{ 8.01065, 18.9662, 1.59175e-06 }, 
	{ 6.65889, 17.7847, -2.1636 }, 
	{ 4.70854, 16.7757, -3.42095 }, 
	{ 3.02145, 16.3591, -4.15867 }, 
	{ 3.02145, 16.3591, 4.15868 }, 
	{ 4.70854, 16.7757, 3.42096 }, 
	{ 6.65888, 17.7847, 2.1636 }, 
	{ 2.91004, 34.2034, 0.94553 }, 
	{ 1.25314, 34.5661, 0.910462 }, 
	{ 3.0598, 34.2034, 1.59175e-06 }, 
	{ 1.47316, 34.5661, -0.478656 }, 
	{ 7.00156, 17.7847, 1.59175e-06 }, 
	{ 5.53522, 16.7757, -1.7985 }, 
	{ 3.63676, 15.9638, -2.64226 }, 
	{ 3.02145, 15.9638, -3.25756 }, 
	{ 3.02145, 15.9638, 3.25756 }, 
	{ 3.63676, 15.9638, 2.64226 }, 
	{ 5.53522, 16.7757, 1.7985 }, 
	{ 1.47316, 34.5661, 0.478659 }, 
	{ 1.54897, 34.5661, 1.59175e-06 }, 
	{ 5.82008, 16.7757, 1.59175e-06 }, 
	{ 4.27527, 15.9638, -1.38912 }, 
	{ 3.02145, 15.6488, -2.19521 }, 
	{ 3.02145, 15.6488, 2.19522 }, 
	{ 4.27527, 15.9638, 1.38912 }, 
	{ 4.49528, 15.9638, 1.59175e-06 }, 
	{ 3.02145, 15.4177, -0.981728 }, 
	{ 3.02145, 15.4177, 0.981731 }, 
	{ 3.0598, 15.3692, 1.59175e-06 }, 
	{ 3.02145, 15.3692, -0.242077 }, 
	{ 3.02145, 15.3692, 0.242077 }, 
	{ 3.02145, 15.36, 1.59175e-06 }
};

static float ZalesakSphereIndices[2202] =
{
	0, 1, 2, 
	3, 1, 0, 
	5, 6, 4, 
	1, 4, 2, 
	2, 4, 6, 
	8, 4, 7, 
	1, 3, 4, 
	7, 4, 3, 
	5, 4, 10, 
	10, 4, 9, 
	6, 5, 12, 
	12, 5, 11, 
	14, 8, 13, 
	13, 8, 7, 
	4, 8, 9, 
	9, 8, 15, 
	11, 5, 16, 
	16, 5, 10, 
	10, 9, 18, 
	18, 9, 17, 
	12, 11, 19, 
	20, 14, 13, 
	8, 14, 15, 
	15, 14, 21, 
	9, 15, 17, 
	17, 15, 22, 
	11, 16, 19, 
	19, 16, 24, 
	16, 23, 24, 
	16, 10, 25, 
	25, 10, 18, 
	18, 17, 27, 
	27, 17, 26, 
	14, 20, 21, 
	20, 28, 21, 
	28, 29, 21, 
	15, 21, 22, 
	22, 21, 30, 
	17, 22, 26, 
	26, 22, 31, 
	23, 16, 32, 
	32, 16, 25, 
	24, 23, 33, 
	25, 18, 34, 
	34, 18, 27, 
	27, 26, 36, 
	36, 26, 35, 
	37, 29, 28, 
	21, 29, 30, 
	30, 29, 38, 
	22, 30, 31, 
	31, 30, 39, 
	26, 31, 35, 
	35, 31, 40, 
	23, 32, 33, 
	33, 32, 41, 
	32, 25, 42, 
	42, 25, 34, 
	34, 27, 43, 
	43, 27, 36, 
	36, 35, 45, 
	45, 35, 44, 
	46, 38, 37, 
	37, 38, 29, 
	30, 38, 39, 
	39, 38, 47, 
	31, 39, 40, 
	40, 39, 48, 
	35, 40, 44, 
	44, 40, 49, 
	32, 42, 41, 
	41, 42, 50, 
	42, 34, 51, 
	51, 34, 43, 
	43, 36, 52, 
	52, 36, 45, 
	45, 44, 54, 
	54, 44, 53, 
	55, 47, 46, 
	46, 47, 38, 
	39, 47, 48, 
	48, 47, 56, 
	40, 48, 49, 
	49, 48, 57, 
	44, 49, 53, 
	53, 49, 58, 
	42, 51, 50, 
	50, 51, 59, 
	51, 43, 60, 
	60, 43, 52, 
	52, 45, 61, 
	61, 45, 54, 
	54, 53, 63, 
	63, 53, 62, 
	64, 56, 55, 
	55, 56, 47, 
	48, 56, 57, 
	57, 56, 65, 
	49, 57, 58, 
	58, 57, 66, 
	53, 58, 62, 
	62, 58, 67, 
	51, 60, 59, 
	59, 60, 68, 
	60, 52, 69, 
	69, 52, 61, 
	61, 54, 70, 
	70, 54, 63, 
	63, 62, 72, 
	72, 62, 71, 
	73, 65, 64, 
	64, 65, 56, 
	57, 65, 66, 
	66, 65, 74, 
	58, 66, 67, 
	67, 66, 75, 
	62, 67, 71, 
	71, 67, 76, 
	60, 69, 68, 
	77, 78, 69, 
	68, 69, 78, 
	69, 61, 79, 
	79, 61, 70, 
	70, 63, 80, 
	80, 63, 72, 
	72, 71, 82, 
	82, 71, 81, 
	65, 73, 74, 
	84, 74, 83, 
	83, 74, 73, 
	66, 74, 75, 
	75, 74, 85, 
	67, 75, 76, 
	76, 75, 86, 
	71, 76, 81, 
	81, 76, 87, 
	77, 69, 88, 
	88, 69, 79, 
	78, 77, 89, 
	79, 70, 90, 
	90, 70, 80, 
	80, 72, 91, 
	91, 72, 82, 
	82, 81, 93, 
	93, 81, 92, 
	94, 84, 83, 
	74, 84, 85, 
	85, 84, 95, 
	75, 85, 86, 
	86, 85, 96, 
	76, 86, 87, 
	87, 86, 97, 
	81, 87, 92, 
	92, 87, 98, 
	89, 77, 99, 
	77, 88, 99, 
	88, 79, 100, 
	100, 79, 90, 
	90, 80, 101, 
	101, 80, 91, 
	91, 82, 102, 
	102, 82, 93, 
	93, 92, 104, 
	104, 92, 103, 
	84, 94, 95, 
	105, 95, 94, 
	85, 95, 96, 
	96, 95, 106, 
	86, 96, 97, 
	97, 96, 107, 
	87, 97, 98, 
	98, 97, 108, 
	92, 98, 103, 
	103, 98, 109, 
	100, 110, 88, 
	99, 88, 110, 
	112, 113, 111, 
	111, 113, 101, 
	101, 113, 90, 
	100, 90, 114, 
	90, 113, 114, 
	101, 91, 115, 
	115, 91, 102, 
	102, 93, 116, 
	116, 93, 104, 
	104, 103, 118, 
	118, 103, 117, 
	119, 106, 105, 
	106, 95, 105, 
	122, 123, 121, 
	123, 107, 121, 
	107, 96, 121, 
	106, 120, 96, 
	96, 120, 121, 
	97, 107, 108, 
	108, 107, 124, 
	98, 108, 109, 
	109, 108, 125, 
	103, 109, 117, 
	117, 109, 126, 
	110, 100, 114, 
	111, 101, 127, 
	127, 101, 115, 
	112, 111, 129, 
	129, 111, 128, 
	115, 102, 130, 
	130, 102, 116, 
	116, 104, 131, 
	131, 104, 118, 
	118, 117, 133, 
	133, 117, 132, 
	106, 119, 120, 
	134, 135, 122, 
	122, 135, 123, 
	107, 123, 124, 
	124, 123, 136, 
	108, 124, 125, 
	125, 124, 137, 
	109, 125, 126, 
	126, 125, 138, 
	117, 126, 132, 
	132, 126, 139, 
	128, 111, 140, 
	140, 111, 127, 
	127, 115, 141, 
	141, 115, 130, 
	129, 128, 143, 
	143, 128, 142, 
	130, 116, 144, 
	144, 116, 131, 
	131, 118, 145, 
	145, 118, 133, 
	133, 132, 147, 
	147, 132, 146, 
	148, 149, 134, 
	134, 149, 135, 
	123, 135, 136, 
	136, 135, 150, 
	124, 136, 137, 
	137, 136, 151, 
	125, 137, 138, 
	138, 137, 152, 
	126, 138, 139, 
	139, 138, 153, 
	132, 139, 146, 
	146, 139, 154, 
	142, 128, 155, 
	155, 128, 140, 
	140, 127, 156, 
	156, 127, 141, 
	141, 130, 157, 
	157, 130, 144, 
	143, 142, 161, 
	142, 162, 161, 
	162, 158, 161, 
	159, 160, 158, 
	158, 160, 161, 
	144, 131, 163, 
	163, 131, 145, 
	145, 133, 164, 
	164, 133, 147, 
	147, 146, 166, 
	166, 146, 165, 
	148, 170, 149, 
	149, 170, 169, 
	169, 170, 168, 
	167, 168, 171, 
	168, 170, 171, 
	135, 149, 150, 
	150, 149, 172, 
	136, 150, 151, 
	151, 150, 173, 
	137, 151, 152, 
	152, 151, 174, 
	138, 152, 153, 
	153, 152, 175, 
	139, 153, 154, 
	154, 153, 176, 
	146, 154, 165, 
	165, 154, 177, 
	162, 142, 178, 
	178, 142, 155, 
	155, 140, 179, 
	179, 140, 156, 
	156, 141, 180, 
	180, 141, 157, 
	157, 144, 181, 
	181, 144, 163, 
	182, 158, 183, 
	183, 158, 162, 
	184, 185, 158, 
	158, 185, 159, 
	159, 186, 160, 
	163, 145, 187, 
	187, 145, 164, 
	164, 147, 188, 
	188, 147, 166, 
	166, 165, 190, 
	190, 165, 189, 
	191, 167, 171, 
	192, 193, 167, 
	167, 193, 168, 
	168, 194, 169, 
	169, 194, 195, 
	149, 169, 172, 
	172, 169, 196, 
	150, 172, 173, 
	173, 172, 197, 
	151, 173, 174, 
	174, 173, 198, 
	152, 174, 175, 
	175, 174, 199, 
	153, 175, 176, 
	176, 175, 200, 
	154, 176, 177, 
	177, 176, 201, 
	165, 177, 189, 
	189, 177, 202, 
	183, 162, 203, 
	203, 162, 178, 
	178, 155, 204, 
	204, 155, 179, 
	179, 156, 205, 
	205, 156, 180, 
	180, 157, 206, 
	206, 157, 181, 
	181, 163, 207, 
	207, 163, 187, 
	208, 182, 209, 
	209, 182, 183, 
	210, 184, 182, 
	182, 184, 158, 
	211, 212, 184, 
	184, 212, 185, 
	186, 159, 213, 
	159, 185, 213, 
	187, 164, 214, 
	214, 164, 188, 
	188, 166, 215, 
	215, 166, 190, 
	190, 189, 216, 
	167, 191, 192, 
	217, 192, 191, 
	218, 219, 192, 
	192, 219, 193, 
	193, 220, 168, 
	168, 220, 194, 
	194, 221, 195, 
	195, 221, 222, 
	169, 195, 196, 
	196, 195, 223, 
	172, 196, 197, 
	197, 196, 224, 
	173, 197, 198, 
	198, 197, 225, 
	174, 198, 199, 
	199, 198, 226, 
	175, 199, 200, 
	200, 199, 227, 
	176, 200, 201, 
	201, 200, 228, 
	177, 201, 202, 
	202, 201, 229, 
	189, 202, 216, 
	209, 183, 230, 
	230, 183, 203, 
	203, 178, 231, 
	231, 178, 204, 
	204, 179, 232, 
	232, 179, 205, 
	205, 180, 233, 
	233, 180, 206, 
	206, 181, 234, 
	234, 181, 207, 
	207, 187, 235, 
	235, 187, 214, 
	236, 208, 237, 
	237, 208, 209, 
	238, 210, 208, 
	208, 210, 182, 
	239, 211, 210, 
	210, 211, 184, 
	241, 242, 211, 
	212, 211, 240, 
	240, 211, 242, 
	212, 243, 185, 
	213, 185, 243, 
	214, 188, 244, 
	244, 188, 215, 
	215, 190, 216, 
	245, 218, 217, 
	218, 192, 217, 
	247, 219, 246, 
	218, 248, 219, 
	246, 219, 248, 
	219, 249, 193, 
	193, 249, 220, 
	220, 250, 194, 
	194, 250, 221, 
	221, 236, 222, 
	222, 236, 237, 
	195, 222, 223, 
	223, 222, 251, 
	196, 223, 224, 
	224, 223, 252, 
	197, 224, 225, 
	225, 224, 253, 
	198, 225, 226, 
	226, 225, 254, 
	199, 226, 227, 
	227, 226, 255, 
	200, 227, 228, 
	228, 227, 256, 
	201, 228, 229, 
	229, 228, 257, 
	202, 229, 216, 
	237, 209, 258, 
	258, 209, 230, 
	230, 203, 259, 
	259, 203, 231, 
	231, 204, 260, 
	260, 204, 232, 
	232, 205, 261, 
	261, 205, 233, 
	233, 206, 262, 
	262, 206, 234, 
	234, 207, 263, 
	263, 207, 235, 
	235, 214, 264, 
	264, 214, 244, 
	265, 238, 236, 
	236, 238, 208, 
	266, 239, 238, 
	238, 239, 210, 
	267, 241, 239, 
	239, 241, 211, 
	243, 212, 240, 
	242, 241, 269, 
	269, 241, 268, 
	244, 215, 216, 
	248, 218, 245, 
	271, 247, 270, 
	270, 247, 246, 
	247, 272, 219, 
	219, 272, 249, 
	249, 273, 220, 
	220, 273, 250, 
	250, 265, 221, 
	221, 265, 236, 
	222, 237, 251, 
	251, 237, 258, 
	223, 251, 252, 
	252, 251, 274, 
	224, 252, 253, 
	253, 252, 275, 
	225, 253, 254, 
	254, 253, 276, 
	226, 254, 255, 
	255, 254, 277, 
	227, 255, 256, 
	256, 255, 278, 
	228, 256, 257, 
	257, 256, 279, 
	229, 257, 216, 
	258, 230, 280, 
	280, 230, 259, 
	259, 231, 281, 
	281, 231, 260, 
	260, 232, 282, 
	282, 232, 261, 
	261, 233, 283, 
	283, 233, 262, 
	262, 234, 284, 
	284, 234, 263, 
	263, 235, 285, 
	285, 235, 264, 
	264, 244, 216, 
	286, 266, 265, 
	265, 266, 238, 
	287, 267, 266, 
	266, 267, 239, 
	288, 268, 267, 
	267, 268, 241, 
	269, 268, 290, 
	290, 268, 289, 
	292, 271, 291, 
	291, 271, 270, 
	271, 293, 247, 
	247, 293, 272, 
	272, 294, 249, 
	249, 294, 273, 
	273, 286, 250, 
	250, 286, 265, 
	251, 258, 274, 
	274, 258, 280, 
	252, 274, 275, 
	275, 274, 295, 
	253, 275, 276, 
	276, 275, 296, 
	254, 276, 277, 
	277, 276, 297, 
	255, 277, 278, 
	278, 277, 298, 
	256, 278, 279, 
	279, 278, 299, 
	257, 279, 216, 
	280, 259, 300, 
	300, 259, 281, 
	281, 260, 301, 
	301, 260, 282, 
	282, 261, 302, 
	302, 261, 283, 
	283, 262, 303, 
	303, 262, 284, 
	284, 263, 304, 
	304, 263, 285, 
	285, 264, 216, 
	305, 287, 286, 
	286, 287, 266, 
	306, 288, 287, 
	287, 288, 267, 
	307, 289, 288, 
	288, 289, 268, 
	290, 289, 309, 
	309, 289, 308, 
	311, 292, 310, 
	310, 292, 291, 
	292, 312, 271, 
	271, 312, 293, 
	293, 313, 272, 
	272, 313, 294, 
	294, 305, 273, 
	273, 305, 286, 
	274, 280, 295, 
	295, 280, 300, 
	275, 295, 296, 
	296, 295, 314, 
	276, 296, 297, 
	297, 296, 315, 
	277, 297, 298, 
	298, 297, 316, 
	278, 298, 299, 
	299, 298, 317, 
	279, 299, 216, 
	300, 281, 318, 
	318, 281, 301, 
	301, 282, 319, 
	319, 282, 302, 
	302, 283, 320, 
	320, 283, 303, 
	303, 284, 321, 
	321, 284, 304, 
	304, 285, 216, 
	322, 306, 305, 
	305, 306, 287, 
	323, 307, 306, 
	306, 307, 288, 
	324, 308, 307, 
	307, 308, 289, 
	309, 308, 326, 
	326, 308, 325, 
	328, 311, 327, 
	327, 311, 310, 
	311, 329, 292, 
	292, 329, 312, 
	312, 330, 293, 
	293, 330, 313, 
	313, 322, 294, 
	294, 322, 305, 
	295, 300, 314, 
	314, 300, 318, 
	296, 314, 315, 
	315, 314, 331, 
	297, 315, 316, 
	316, 315, 332, 
	298, 316, 317, 
	317, 316, 333, 
	299, 317, 216, 
	318, 301, 334, 
	334, 301, 319, 
	319, 302, 335, 
	335, 302, 320, 
	320, 303, 336, 
	336, 303, 321, 
	321, 304, 216, 
	337, 323, 322, 
	322, 323, 306, 
	338, 324, 323, 
	323, 324, 307, 
	339, 325, 324, 
	324, 325, 308, 
	326, 325, 340, 
	341, 328, 327, 
	328, 342, 311, 
	311, 342, 329, 
	329, 343, 312, 
	312, 343, 330, 
	330, 337, 313, 
	313, 337, 322, 
	314, 318, 331, 
	331, 318, 334, 
	315, 331, 332, 
	332, 331, 344, 
	316, 332, 333, 
	333, 332, 345, 
	317, 333, 216, 
	334, 319, 346, 
	346, 319, 335, 
	335, 320, 347, 
	347, 320, 336, 
	336, 321, 216, 
	348, 338, 337, 
	337, 338, 323, 
	349, 339, 338, 
	338, 339, 324, 
	350, 351, 339, 
	351, 340, 339, 
	340, 325, 339, 
	353, 342, 352, 
	352, 342, 341, 
	342, 328, 341, 
	342, 354, 329, 
	329, 354, 343, 
	343, 348, 330, 
	330, 348, 337, 
	331, 334, 344, 
	344, 334, 346, 
	332, 344, 345, 
	345, 344, 355, 
	333, 345, 216, 
	346, 335, 356, 
	356, 335, 347, 
	347, 336, 216, 
	357, 349, 348, 
	348, 349, 338, 
	358, 350, 349, 
	349, 350, 339, 
	351, 350, 359, 
	360, 353, 352, 
	353, 361, 342, 
	342, 361, 354, 
	354, 357, 343, 
	343, 357, 348, 
	344, 346, 355, 
	355, 346, 356, 
	345, 355, 216, 
	356, 347, 216, 
	362, 358, 357, 
	357, 358, 349, 
	350, 358, 359, 
	359, 358, 363, 
	364, 361, 360, 
	360, 361, 353, 
	361, 362, 354, 
	354, 362, 357, 
	355, 356, 216, 
	358, 362, 363, 
	365, 366, 362, 
	363, 362, 366, 
	361, 364, 362, 
	365, 362, 367, 
	367, 362, 364, 
	366, 365, 368, 
	368, 365, 367, 
	170, 148, 161, 
	148, 134, 161, 
	134, 122, 161, 
	122, 121, 161, 
	161, 121, 143, 
	143, 121, 129, 
	113, 112, 121, 
	121, 112, 129, 
	161, 160, 170, 
	170, 160, 171, 
	160, 186, 171, 
	186, 213, 171, 
	213, 243, 171, 
	243, 240, 171, 
	240, 242, 171, 
	242, 269, 171, 
	269, 290, 171, 
	290, 309, 171, 
	309, 326, 171, 
	326, 340, 171, 
	340, 351, 171, 
	351, 359, 171, 
	359, 363, 171, 
	363, 366, 171, 
	366, 368, 171, 
	368, 367, 171, 
	367, 364, 171, 
	364, 360, 171, 
	360, 352, 171, 
	352, 341, 171, 
	341, 327, 171, 
	327, 310, 171, 
	310, 291, 171, 
	291, 270, 171, 
	270, 246, 171, 
	171, 246, 191, 
	191, 246, 217, 
	246, 248, 217, 
	248, 245, 217, 
	121, 120, 113, 
	113, 120, 114, 
	120, 119, 114, 
	119, 105, 114, 
	105, 94, 114, 
	94, 83, 114, 
	83, 73, 114, 
	73, 64, 114, 
	64, 55, 114, 
	55, 46, 114, 
	46, 37, 114, 
	37, 28, 114, 
	28, 20, 114, 
	20, 13, 114, 
	13, 7, 114, 
	7, 3, 114, 
	3, 0, 114, 
	0, 2, 114, 
	2, 6, 114, 
	6, 12, 114, 
	12, 19, 114, 
	19, 24, 114, 
	24, 33, 114, 
	33, 41, 114, 
	41, 50, 114, 
	50, 59, 114, 
	59, 68, 114, 
	114, 68, 110, 
	110, 68, 99, 
	68, 78, 99, 
	78, 89, 99
};

BORA_NAMESPACE_END

#endif

