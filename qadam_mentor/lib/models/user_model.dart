import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  final String uid;
  final String email;
  final String name;
  final String role; // 'student' | 'mentor'
  final String? photoUrl;
  final String? bio;
  final String? specialty; // for mentors
  final String? year;      // for students (1-4 курс)
  final String? phone;
  final DateTime createdAt;

  const UserModel({
    required this.uid,
    required this.email,
    required this.name,
    required this.role,
    this.photoUrl,
    this.bio,
    this.specialty,
    this.year,
    this.phone,
    required this.createdAt,
  });

  factory UserModel.fromMap(Map<String, dynamic> map) {
    return UserModel(
      uid: map['uid'] as String,
      email: map['email'] as String,
      name: map['name'] as String,
      role: map['role'] as String? ?? 'student',
      photoUrl: map['photoUrl'] as String?,
      bio: map['bio'] as String?,
      specialty: map['specialty'] as String?,
      year: map['year'] as String?,
      phone: map['phone'] as String?,
      createdAt: map['createdAt'] is Timestamp
          ? (map['createdAt'] as Timestamp).toDate()
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() => {
        'uid': uid,
        'email': email,
        'name': name,
        'role': role,
        'photoUrl': photoUrl,
        'bio': bio,
        'specialty': specialty,
        'year': year,
        'phone': phone,
        'createdAt': Timestamp.fromDate(createdAt),
      };

  UserModel copyWith({
    String? name,
    String? photoUrl,
    String? bio,
    String? specialty,
    String? year,
    String? phone,
  }) {
    return UserModel(
      uid: uid,
      email: email,
      name: name ?? this.name,
      role: role,
      photoUrl: photoUrl ?? this.photoUrl,
      bio: bio ?? this.bio,
      specialty: specialty ?? this.specialty,
      year: year ?? this.year,
      phone: phone ?? this.phone,
      createdAt: createdAt,
    );
  }
}
