//! CoreAudio live audio capture for macOS.
//!
//! Enumerates input devices, captures audio via AudioUnit (HAL Input),
//! and resamples to 16 kHz mono f32 for the ASR pipeline.

use coreaudio_sys::*;
use std::ffi::CStr;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::sync::mpsc;

// ========================================================================
// Device Enumeration
// ========================================================================

/// An audio input device.
pub struct AudioDevice {
    pub id: AudioDeviceID,
    pub name: String,
    pub input_channels: u32,
}

/// Get the list of audio input devices.
pub fn list_input_devices() -> Vec<AudioDevice> {
    let mut devices = Vec::new();

    // Get all audio devices
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut data_size: u32 = 0;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
        )
    };
    if status != 0 || data_size == 0 {
        return devices;
    }

    let device_count = data_size as usize / mem::size_of::<AudioDeviceID>();
    let mut device_ids = vec![0u32; device_count];

    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
            device_ids.as_mut_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return devices;
    }

    for &device_id in &device_ids {
        // Check if device has input channels
        let input_channels = get_input_channel_count(device_id);
        if input_channels == 0 {
            continue;
        }

        let name = get_device_name(device_id);
        devices.push(AudioDevice {
            id: device_id,
            name,
            input_channels,
        });
    }

    devices
}

fn get_device_name(device_id: AudioDeviceID) -> String {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyDeviceNameCFString,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut name_ref: CFStringRef = ptr::null();
    let mut data_size = mem::size_of::<CFStringRef>() as u32;

    let status = unsafe {
        AudioObjectGetPropertyData(
            device_id,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
            &mut name_ref as *mut _ as *mut c_void,
        )
    };

    if status != 0 || name_ref.is_null() {
        return format!("Device {}", device_id);
    }

    // Convert CFString to Rust String
    let c_str = unsafe { CFStringGetCStringPtr(name_ref, kCFStringEncodingUTF8) };
    let name = if !c_str.is_null() {
        unsafe { CStr::from_ptr(c_str) }
            .to_string_lossy()
            .into_owned()
    } else {
        // Fallback: use CFStringGetCString
        let mut buf = [0i8; 256];
        let ok = unsafe {
            CFStringGetCString(
                name_ref,
                buf.as_mut_ptr(),
                buf.len() as CFIndex,
                kCFStringEncodingUTF8,
            )
        };
        if ok != 0 {
            unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("Device {}", device_id)
        }
    };

    unsafe { CFRelease(name_ref as *const c_void) };
    name
}

fn get_input_channel_count(device_id: AudioDeviceID) -> u32 {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: kAudioObjectPropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut data_size: u32 = 0;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            device_id,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
        )
    };
    if status != 0 || data_size == 0 {
        return 0;
    }

    let mut buf = vec![0u8; data_size as usize];
    let status = unsafe {
        AudioObjectGetPropertyData(
            device_id,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
            buf.as_mut_ptr() as *mut c_void,
        )
    };
    if status != 0 {
        return 0;
    }

    let buffer_list = unsafe { &*(buf.as_ptr() as *const AudioBufferList) };
    let mut total_channels: u32 = 0;

    let n_buffers = buffer_list.mNumberBuffers as usize;
    if n_buffers == 0 {
        return 0;
    }

    // Access the variable-length mBuffers array
    let buffers_ptr = &buffer_list.mBuffers as *const AudioBuffer;
    for i in 0..n_buffers {
        let ab = unsafe { &*buffers_ptr.add(i) };
        total_channels += ab.mNumberChannels;
    }

    total_channels
}

/// Find an input device by name (case-insensitive substring match).
pub fn find_device_by_name(name: &str) -> Option<AudioDevice> {
    let name_lower = name.to_lowercase();
    let devices = list_input_devices();
    devices
        .into_iter()
        .find(|d| d.name.to_lowercase().contains(&name_lower))
}

/// Get the default input device.
pub fn default_input_device() -> Option<AudioDeviceID> {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut device_id: AudioDeviceID = 0;
    let mut data_size = mem::size_of::<AudioDeviceID>() as u32;

    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
            &mut device_id as *mut _ as *mut c_void,
        )
    };

    if status != 0 || device_id == kAudioObjectUnknown {
        None
    } else {
        Some(device_id)
    }
}

/// Print all input devices to stderr.
pub fn print_devices() {
    let devices = list_input_devices();
    if devices.is_empty() {
        eprintln!("No audio input devices found.");
        return;
    }

    let default_id = default_input_device();

    eprintln!("Audio input devices:\n");
    for d in &devices {
        let marker = if Some(d.id) == default_id { " (default)" } else { "" };
        eprintln!("  {:30} {} ch{}", d.name, d.input_channels, marker);
    }
    eprintln!();
}

// ========================================================================
// Audio Capture
// ========================================================================

/// Capture handle — drop to stop capture.
pub struct CaptureHandle {
    audio_unit: AudioUnit,
    _state: Box<CaptureCallbackState>,
}

/// Callback state passed to the AudioUnit render callback via ref_con.
struct CaptureCallbackState {
    tx: mpsc::Sender<Vec<f32>>,
    audio_unit: AudioUnit,
}

/// Actual render callback using CaptureCallbackState.
unsafe extern "C" fn render_callback(
    in_ref_con: *mut c_void,
    io_action_flags: *mut AudioUnitRenderActionFlags,
    in_time_stamp: *const AudioTimeStamp,
    in_bus_number: u32,
    in_number_frames: u32,
    _io_data: *mut AudioBufferList,
) -> OSStatus {
    let state = &*(in_ref_con as *const CaptureCallbackState);

    let n = in_number_frames as usize;
    let mut samples = vec![0f32; n];

    let buffer = AudioBuffer {
        mNumberChannels: 1,
        mDataByteSize: (n * mem::size_of::<f32>()) as u32,
        mData: samples.as_mut_ptr() as *mut c_void,
    };

    let mut buffer_list = AudioBufferList {
        mNumberBuffers: 1,
        mBuffers: [buffer],
    };

    let status = AudioUnitRender(
        state.audio_unit,
        io_action_flags,
        in_time_stamp,
        in_bus_number,
        in_number_frames,
        &mut buffer_list,
    );

    if status != 0 {
        return status;
    }

    let _ = state.tx.send(samples);
    0
}

/// Start capturing audio from a device. Returns a channel receiver for audio
/// chunks (f32, mono, at device sample rate) and a handle to stop capture.
pub fn start_capture(
    device_id: AudioDeviceID,
) -> Result<(mpsc::Receiver<Vec<f32>>, CaptureHandle, f64), String> {
    // Get device's native sample rate
    let sample_rate = get_device_sample_rate(device_id)?;

    // Create AUHAL AudioUnit
    let comp_desc = AudioComponentDescription {
        componentType: kAudioUnitType_Output,
        componentSubType: kAudioUnitSubType_HALOutput,
        componentManufacturer: kAudioUnitManufacturer_Apple,
        componentFlags: 0,
        componentFlagsMask: 0,
    };

    let component = unsafe { AudioComponentFindNext(ptr::null_mut(), &comp_desc) };
    if component.is_null() {
        return Err("Cannot find HAL Output AudioComponent".into());
    }

    let mut audio_unit: AudioUnit = ptr::null_mut();
    let status = unsafe { AudioComponentInstanceNew(component, &mut audio_unit) };
    if status != 0 {
        return Err(format!("AudioComponentInstanceNew failed: {}", status));
    }

    // Enable input on bus 1 (input element)
    let enable_io: u32 = 1;
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1, // input element
            &enable_io as *const _ as *const c_void,
            mem::size_of::<u32>() as u32,
        )
    };
    if status != 0 {
        return Err(format!("Enable input IO failed: {}", status));
    }

    // Disable output on bus 0 (output element)
    let disable_io: u32 = 0;
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0, // output element
            &disable_io as *const _ as *const c_void,
            mem::size_of::<u32>() as u32,
        )
    };
    if status != 0 {
        return Err(format!("Disable output IO failed: {}", status));
    }

    // Set the input device
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &device_id as *const _ as *const c_void,
            mem::size_of::<AudioDeviceID>() as u32,
        )
    };
    if status != 0 {
        return Err(format!("Set current device failed: {}", status));
    }

    // Set output format of bus 1 (what we read from the callback):
    // Float32, mono, device sample rate
    let stream_format = AudioStreamBasicDescription {
        mSampleRate: sample_rate,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
        mBytesPerPacket: 4,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4,
        mChannelsPerFrame: 1,
        mBitsPerChannel: 32,
        mReserved: 0,
    };

    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1, // output scope of input element = data we receive
            &stream_format as *const _ as *const c_void,
            mem::size_of::<AudioStreamBasicDescription>() as u32,
        )
    };
    if status != 0 {
        return Err(format!("Set stream format failed: {}", status));
    }

    // Create channel and state
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let state = Box::new(CaptureCallbackState {
        tx,
        audio_unit,
    });

    // Set input callback
    let callback_struct = AURenderCallbackStruct {
        inputProc: Some(render_callback),
        inputProcRefCon: &*state as *const CaptureCallbackState as *mut c_void,
    };

    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            0,
            &callback_struct as *const _ as *const c_void,
            mem::size_of::<AURenderCallbackStruct>() as u32,
        )
    };
    if status != 0 {
        return Err(format!("Set input callback failed: {}", status));
    }

    // Initialize and start
    let status = unsafe { AudioUnitInitialize(audio_unit) };
    if status != 0 {
        return Err(format!("AudioUnitInitialize failed: {}", status));
    }

    let status = unsafe { AudioOutputUnitStart(audio_unit) };
    if status != 0 {
        return Err(format!("AudioOutputUnitStart failed: {}", status));
    }

    let handle = CaptureHandle {
        audio_unit,
        _state: state,
    };

    Ok((rx, handle, sample_rate))
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        unsafe {
            AudioOutputUnitStop(self.audio_unit);
            AudioUnitUninitialize(self.audio_unit);
            AudioComponentInstanceDispose(self.audio_unit);
        }
    }
}

fn get_device_sample_rate(device_id: AudioDeviceID) -> Result<f64, String> {
    let property_address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut sample_rate: f64 = 0.0;
    let mut data_size = mem::size_of::<f64>() as u32;

    let status = unsafe {
        AudioObjectGetPropertyData(
            device_id,
            &property_address,
            0,
            ptr::null(),
            &mut data_size,
            &mut sample_rate as *mut _ as *mut c_void,
        )
    };

    if status != 0 {
        return Err(format!(
            "Cannot get sample rate for device {}: error {}",
            device_id, status
        ));
    }

    Ok(sample_rate)
}
