/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///
/// ```java
/// public class QAsrEngine {
///     static { System.loadLibrary("qwen_asr"); }
///     private long nativeHandle;
///     public native boolean loadModel(String modelDir, int nThreads);
///     public native String transcribePcm(float[] samples);
///     public native String transcribeWav(byte[] wavData);
///     public native void setSegmentSec(float sec);
///     public native void setLanguage(String language);
///     public native void free();
/// }
/// ```

use std::os::raw::{c_char, c_void};

// JNI types
type JNIEnv = *mut c_void;
type JClass = *mut c_void;
type JString = *mut c_void;
type JObject = *mut c_void;
type JFloatArray = *mut c_void;
type JByteArray = *mut c_void;
type JLong = i64;
type JInt = i32;
type JFloat = f32;
type JBoolean = u8;

// JNI function signatures we need
extern "C" {
    fn __android_log_write(prio: i32, tag: *const c_char, text: *const c_char) -> i32;
}

// JNI string helpers (accessed via function pointers in JNIEnv)
// These are simplified - a real implementation would use jni-rs or manual vtable access.
// For now, we use the C-FFI API underneath.

/// Load model. Returns native handle as jlong.
#[no_mangle]
pub unsafe extern "C" fn Java_com_qwenasr_QAsrEngine_nativeLoadModel(
    env: JNIEnv,
    _obj: JObject,
    model_dir: JString,
    n_threads: JInt,
) -> JLong {
    // Use the C-FFI API for the actual implementation
    let c_api_engine = crate::c_api::qwen_asr_load_model(
        jstring_to_cstr(env, model_dir),
        n_threads,
        0, // silent
    );
    c_api_engine as JLong
}

/// Transcribe PCM float array.
#[no_mangle]
pub unsafe extern "C" fn Java_com_qwenasr_QAsrEngine_nativeTranscribePcm(
    env: JNIEnv,
    _obj: JObject,
    handle: JLong,
    samples: JFloatArray,
    n_samples: JInt,
) -> JString {
    if handle == 0 {
        return std::ptr::null_mut();
    }

    let samples_ptr = jfloat_array_get(env, samples);
    if samples_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = crate::c_api::qwen_asr_transcribe_pcm(
        handle as *mut crate::c_api::QwenAsrEngine,
        samples_ptr,
        n_samples,
    );

    jfloat_array_release(env, samples, samples_ptr);

    if result.is_null() {
        return std::ptr::null_mut();
    }

    let jstr = new_jstring(env, result);
    crate::c_api::qwen_asr_free_string(result);
    jstr
}

/// Transcribe WAV byte array.
#[no_mangle]
pub unsafe extern "C" fn Java_com_qwenasr_QAsrEngine_nativeTranscribeWav(
    env: JNIEnv,
    _obj: JObject,
    handle: JLong,
    wav_data: JByteArray,
    wav_len: JInt,
) -> JString {
    if handle == 0 {
        return std::ptr::null_mut();
    }

    let data_ptr = jbyte_array_get(env, wav_data);
    if data_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = crate::c_api::qwen_asr_transcribe_wav_buffer(
        handle as *mut crate::c_api::QwenAsrEngine,
        data_ptr as *const u8,
        wav_len,
    );

    jbyte_array_release(env, wav_data, data_ptr);

    if result.is_null() {
        return std::ptr::null_mut();
    }

    let jstr = new_jstring(env, result);
    crate::c_api::qwen_asr_free_string(result);
    jstr
}

/// Free engine.
#[no_mangle]
pub unsafe extern "C" fn Java_com_qwenasr_QAsrEngine_nativeFree(
    _env: JNIEnv,
    _obj: JObject,
    handle: JLong,
) {
    if handle != 0 {
        crate::c_api::qwen_asr_free(handle as *mut crate::c_api::QwenAsrEngine);
    }
}

/// Set segment seconds.
#[no_mangle]
pub unsafe extern "C" fn Java_com_qwenasr_QAsrEngine_nativeSetSegmentSec(
    _env: JNIEnv,
    _obj: JObject,
    handle: JLong,
    sec: JFloat,
) {
    if handle != 0 {
        crate::c_api::qwen_asr_set_segment_sec(handle as *mut crate::c_api::QwenAsrEngine, sec);
    }
}

// ========================================================================
// JNI helper stubs - these use raw JNIEnv vtable access
// In production, use the `jni` crate. These are minimal stubs for compilation.
// ========================================================================

unsafe fn jstring_to_cstr(_env: JNIEnv, _jstr: JString) -> *const c_char {
    // Real implementation: (*(*env)).GetStringUTFChars(env, jstr, null)
    // Stub returns null - will need jni crate for real Android builds
    std::ptr::null()
}

unsafe fn jfloat_array_get(_env: JNIEnv, _arr: JFloatArray) -> *const f32 {
    std::ptr::null()
}

unsafe fn jfloat_array_release(_env: JNIEnv, _arr: JFloatArray, _ptr: *const f32) {}

unsafe fn jbyte_array_get(_env: JNIEnv, _arr: JByteArray) -> *const i8 {
    std::ptr::null()
}

unsafe fn jbyte_array_release(_env: JNIEnv, _arr: JByteArray, _ptr: *const i8) {}

unsafe fn new_jstring(_env: JNIEnv, _cstr: *const c_char) -> JString {
    std::ptr::null_mut()
}
