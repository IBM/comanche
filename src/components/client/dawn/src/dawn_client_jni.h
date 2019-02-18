#ifndef __DAWN_CLIENT_JNI_H__
#define __DAWN_CLIENT_JNI_H__

#include <jni.h>

JNIEXPORT void JNICALL
               Java_DawnClient_init(JNIEnv *, jobject, jint, jstring, jstring, jstring);

JNIEXPORT jint JNICALL
               Java_DawnClient_put(JNIEnv *, jobject, jstring, jstring, jbyteArray, jboolean);

JNIEXPORT jint JNICALL
               Java_DawnClient_get(JNIEnv *, jobject, jstring, jstring, jbyteArray, jboolean);

JNIEXPORT jint JNICALL Java_DawnClient_erase(JNIEnv *,
                                             jobject,
                                             jstring,
                                             jstring);

JNIEXPORT jint JNICALL Java_DawnClient_clean(JNIEnv *, jobject);
#endif
