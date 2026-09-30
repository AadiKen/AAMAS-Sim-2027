import type {Quat,Vec3} from './spatial';

export const POSITION_NED='position_ned_m';
export const BODY_VELOCITY='nu_body';
export type ReplayState={position_display_m:Vec3;q_body_to_ned:Quat;[BODY_VELOCITY]:number[]};
