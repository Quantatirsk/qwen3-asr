class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(2560);
    this.used = 0;
    this.stopped = false;
    this.port.onmessage = ({data}) => {
      if (data === 'stop') {
        this.stopped = true;
        if (this.used) this.port.postMessage(this.frame.slice(0, this.used).buffer);
        this.port.postMessage('stopped');
      }
    };
  }
  process(inputs) {
    if (this.stopped) return false;
    const audio = inputs[0]?.[0];
    if (!audio) return true;
    for (const value of audio) {
      const x = Math.max(-1, Math.min(1, value));
      this.frame[this.used++] = Math.round(x * (x < 0 ? 32768 : 32767));
      if (this.used === this.frame.length) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Int16Array(2560);
        this.used = 0;
      }
    }
    return true;
  }
}
registerProcessor('recorder', Recorder);
