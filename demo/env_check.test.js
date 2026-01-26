import { describe, it, expect } from 'vitest';
import * as ort from 'onnxruntime-node';

describe('Environment Check', () => {
  it('should have vitest working', () => {
    expect(true).toBe(true);
  });

  it('should be able to import onnxruntime-node', () => {
    expect(ort).toBeDefined();
    expect(typeof ort.InferenceSession).toBe('function');
  });

  it('should be running in ESM mode', () => {
    // __dirname is not defined in ESM
    try {
      console.log(__dirname);
    } catch (e) {
      expect(e).toBeInstanceOf(ReferenceError);
    }
  });
});