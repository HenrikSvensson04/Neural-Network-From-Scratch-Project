/* tslint:disable */
/* eslint-disable */
/**
* @returns {string}
*/
export function str(): string;
/**
* @param {NeuralWrapper} wrapper
* @returns {string}
*/
export function echo(wrapper: NeuralWrapper): string;
/**
* @param {NeuralWrapper} wrapper
* @returns {string}
*/
export function serialize_neural_wrapper(wrapper: NeuralWrapper): string;
/**
*/
export class NeuralWrapper {
  free(): void;
/**
* @param {string} structure_network
* @param {string} type_of_creation
* @param {string} json
*/
  constructor(structure_network: string, type_of_creation: string, json: string);
/**
*/
  initalize_traning_handeler(): void;
/**
* @param {Float32Array} input
* @param {Float32Array} correct_output
*/
  insert_traning_data(input: Float32Array, correct_output: Float32Array): void;
/**
* @param {number} epochs
*/
  train(epochs: number): void;
/**
* TODO: Make this function more safe!
* @param {Float32Array} input
* @returns {Float32Array}
*/
  get_output(input: Float32Array): Float32Array;
/**
* @returns {number}
*/
  get(): number;
/**
* @returns {string}
*/
  get_json_serialized(): string;
/**
* @returns {number}
*/
  get_cost(): number;
/**
* @param {number} learning_rate
*/
  set_learning_rate(learning_rate: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_neuralwrapper_free: (a: number) => void;
  readonly neuralwrapper_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly neuralwrapper_initalize_traning_handeler: (a: number) => void;
  readonly neuralwrapper_insert_traning_data: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly neuralwrapper_train: (a: number, b: number) => void;
  readonly neuralwrapper_get_output: (a: number, b: number, c: number, d: number) => void;
  readonly neuralwrapper_get: (a: number) => number;
  readonly neuralwrapper_get_json_serialized: (a: number, b: number) => void;
  readonly neuralwrapper_get_cost: (a: number) => number;
  readonly neuralwrapper_set_learning_rate: (a: number, b: number) => void;
  readonly str: (a: number) => void;
  readonly echo: (a: number, b: number) => void;
  readonly serialize_neural_wrapper: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
